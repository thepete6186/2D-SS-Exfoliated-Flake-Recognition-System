"""
Thin orchestration layer for camera + stage coordination.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from camera.coordinate_mapper import CoordinateMapper
from stage.base import StageError, StageNotConnectedError

logger = logging.getLogger(__name__)


@dataclass
class AlignmentSettings:
    """Persisted settings for click-to-move alignment."""

    stage_profile: str = "zolix"
    calibration_name: str = "default"
    invert_x: bool = False
    invert_y: bool = False
    flip_xy: bool = False
    xy_um_per_step: float = 0.5
    jog_steps: int = 1000
    use_simulated_stage: bool = True
    stage_port: str = "COM3"
    stage_slave: int = 1


class AlignmentSettingsStore:
    """JSON persistence for alignment settings."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> AlignmentSettings:
        if not self.path.exists():
            return AlignmentSettings()
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return AlignmentSettings(**raw)
        except Exception as exc:
            logger.warning("Failed to load alignment settings: %s", exc)
            return AlignmentSettings()

    def save(self, settings: AlignmentSettings) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(asdict(settings), f, indent=2)
        except Exception as exc:
            logger.warning("Failed to save alignment settings: %s", exc)


class CameraSession:
    """Background capture service around an existing camera backend object."""

    def __init__(
        self,
        camera_factory: Callable[[], Any],
        on_frame: Optional[Callable[[Any], None]] = None,
    ) -> None:
        self._camera_factory = camera_factory
        self._on_frame = on_frame
        self._camera: Optional[Any] = None
        self._connected = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest_frame = None

    def connect(self) -> bool:
        if self._connected and self._camera is not None:
            return True
        camera = self._camera_factory()
        if not camera.connect():
            return False
        self._camera = camera
        self._connected = True
        return True

    def disconnect(self) -> None:
        self.stop()
        if self._camera is not None:
            try:
                self._camera.disconnect()
            except Exception:
                pass
        self._camera = None
        self._connected = False
        with self._lock:
            self._latest_frame = None

    def start(self) -> None:
        if not self._connected or self._camera is None or self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def _capture_loop(self) -> None:
        assert self._camera is not None
        while not self._stop.is_set():
            try:
                frame = self._camera.capture()
                if frame is None:
                    time.sleep(0.01)
                    continue
                with self._lock:
                    self._latest_frame = frame
                if self._on_frame:
                    self._on_frame(frame)
            except Exception as exc:
                logger.warning("CameraSession capture loop error: %s", exc)
                time.sleep(0.05)

    def get_latest_frame(self):
        with self._lock:
            return self._latest_frame

    def set_exposure(self, exposure_us: float) -> None:
        if self._camera is not None:
            self._camera.set_exposure(exposure_us)

    def set_gain(self, gain_db: float) -> None:
        if self._camera is not None:
            self._camera.set_gain(gain_db)

    @property
    def is_connected(self) -> bool:
        return self._connected and self._camera is not None


class StageSession:
    """Lifecycle wrapper around a stage implementation."""

    def __init__(self, stage_factory: Callable[[], Any]):
        self._stage_factory = stage_factory
        self._stage = None

    def connect(self) -> None:
        if self._stage is not None and self._stage.is_connected:
            return
        self._stage = self._stage_factory()
        self._stage.connect()

    def disconnect(self) -> None:
        if self._stage is not None:
            try:
                self._stage.disconnect()
            finally:
                self._stage = None

    @property
    def is_connected(self) -> bool:
        return bool(self._stage is not None and self._stage.is_connected)

    def home(self, axis: str = "all") -> None:
        self._require_stage()
        self._stage.home(axis, wait=True)

    def jog(self, axis: str, steps: int) -> None:
        self._require_stage()
        self._stage.move_relative(axis, int(steps), wait=True)

    def move_xy(self, steps_x: int, steps_y: int) -> None:
        self._require_stage()
        if int(steps_x) != 0:
            self._stage.move_relative("x", int(steps_x), wait=True)
        if int(steps_y) != 0:
            self._stage.move_relative("y", int(steps_y), wait=True)

    def get_status(self) -> Dict[str, Any]:
        self._require_stage()
        return self._stage.get_status()

    def _require_stage(self) -> None:
        if self._stage is None or not self._stage.is_connected:
            raise StageNotConnectedError("Stage is not connected")


class AlignmentMapper:
    """Pixel -> stage command policy wrapper."""

    def __init__(self, coordinate_mapper: CoordinateMapper):
        self.coordinate_mapper = coordinate_mapper

    def update_um_per_step(self, stage_profile: str, um_per_step: float) -> None:
        if stage_profile not in self.coordinate_mapper.stage_config:
            self.coordinate_mapper.stage_config[stage_profile] = {}
        self.coordinate_mapper.stage_config[stage_profile]["xy_um_per_step"] = float(um_per_step)

    def pixel_to_stage_command(
        self,
        px: int,
        py: int,
        image_size: Tuple[int, int],
        settings: AlignmentSettings,
    ) -> Dict[str, Any]:
        self.update_um_per_step(settings.stage_profile, settings.xy_um_per_step)
        return self.coordinate_mapper.pixel_to_stage_commands(
            px,
            py,
            image_size,
            stage=settings.stage_profile,
            invert_x=settings.invert_x,
            invert_y=settings.invert_y,
            flip_xy=settings.flip_xy,
        )


@dataclass
class _Intent:
    kind: str
    payload: Dict[str, Any]


class CameraStageOrchestrator:
    """Single-worker queued stage command orchestrator."""

    FRAME_UPDATED = "FRAME_UPDATED"
    STAGE_STATUS = "STAGE_STATUS"
    STAGE_ERROR = "STAGE_ERROR"
    MOVE_STARTED = "MOVE_STARTED"
    MOVE_DONE = "MOVE_DONE"

    def __init__(
        self,
        stage_session: StageSession,
        alignment_mapper: AlignmentMapper,
        settings_provider: Callable[[], AlignmentSettings],
        on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> None:
        self.stage_session = stage_session
        self.alignment_mapper = alignment_mapper
        self.settings_provider = settings_provider
        self.on_event = on_event

        self._queue: "queue.Queue[_Intent]" = queue.Queue()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def close(self) -> None:
        self._stop.set()
        self._queue.put(_Intent("STOP", {}))
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)

    def connect_stage(self) -> None:
        try:
            self.stage_session.connect()
            self._publish(self.STAGE_STATUS, {"status": self.stage_session.get_status()})
        except StageError as exc:
            self._publish(self.STAGE_ERROR, {"error": str(exc)})

    def disconnect_stage(self) -> None:
        self.stage_session.disconnect()
        self._publish(self.STAGE_STATUS, {"status": None})

    def enqueue_home(self, axis: str = "all") -> None:
        self._queue.put(_Intent("HOME", {"axis": axis}))

    def enqueue_jog(self, axis: str, steps: int) -> None:
        self._queue.put(_Intent("JOG", {"axis": axis, "steps": int(steps)}))

    def enqueue_click_move(self, px: int, py: int, image_size: Tuple[int, int]) -> None:
        self._queue.put(
            _Intent("MOVE_CLICK", {"px": int(px), "py": int(py), "image_size": image_size})
        )

    def enqueue_status(self) -> None:
        self._queue.put(_Intent("STATUS", {}))

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            intent = self._queue.get()
            if intent.kind == "STOP":
                return
            try:
                if intent.kind == "HOME":
                    self._publish(self.MOVE_STARTED, {"kind": "HOME", **intent.payload})
                    self.stage_session.home(intent.payload["axis"])
                    self._publish(self.MOVE_DONE, {"kind": "HOME", **intent.payload})
                elif intent.kind == "JOG":
                    self._publish(self.MOVE_STARTED, {"kind": "JOG", **intent.payload})
                    self.stage_session.jog(intent.payload["axis"], intent.payload["steps"])
                    self._publish(self.MOVE_DONE, {"kind": "JOG", **intent.payload})
                elif intent.kind == "MOVE_CLICK":
                    self._publish(self.MOVE_STARTED, {"kind": "MOVE_CLICK", **intent.payload})
                    settings = self.settings_provider()
                    cmd = self.alignment_mapper.pixel_to_stage_command(
                        intent.payload["px"],
                        intent.payload["py"],
                        intent.payload["image_size"],
                        settings,
                    )
                    self.stage_session.move_xy(cmd["steps_x"], cmd["steps_y"])
                    self._publish(self.MOVE_DONE, {"kind": "MOVE_CLICK", "command": cmd})
                elif intent.kind == "STATUS":
                    pass

                self._publish(self.STAGE_STATUS, {"status": self.stage_session.get_status()})
            except StageError as exc:
                self._publish(self.STAGE_ERROR, {"error": str(exc), "intent": intent.kind})

    def _publish(self, event_type: str, payload: Dict[str, Any]) -> None:
        if self.on_event is not None:
            self.on_event(event_type, payload)

