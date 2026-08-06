"""Direct U3V (USB3 Vision) camera driver for Zeiss Axiocam 208 via pyusb."""
import usb.core
import usb.util
import struct
import numpy as np
import cv2
import logging
import time
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

ZEISS_VID = 0x0758
ZEISS_PID = 0x6002

CTRL_OUT = 0x02
CTRL_IN = 0x82
STREAM_IN = 0x81

READREG = 0x0080
WRITEREG = 0x0082
CONTROL = 0x0088


class U3VCamera:
    """Direct USB3 Vision camera driver using pyusb + GVCP protocol."""

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.dev = None
        self.connected = False
        self.backend = "u3v"
        self.settings: Dict[str, Any] = {
            "exposure_us": 10000.0,
            "gain_db": 0.0,
            "resolution": (5472, 3648),
        }
        self._req_id = 1

    def connect(self) -> bool:
        try:
            self.dev = usb.core.find(idVendor=ZEISS_VID, idProduct=ZEISS_PID)
            if self.dev is None:
                logger.error("U3V: camera not found on USB bus")
                return False

            try:
                if self.dev.is_kernel_driver_active(0):
                    self.dev.detach_kernel_driver(0)
            except Exception:
                pass

            usb.util.claim_interface(self.dev, 0)

            # Test communication - try GVCP READREG
            magic = self.read_register(0x0000)
            if magic is None:
                logger.error("U3V: camera not responding to GVCP")
                usb.util.release_interface(self.dev, 0)
                return False

            self.connected = True
            logger.info(f"U3V: connected, magic=0x{magic:08X}")
            return True
        except Exception as e:
            logger.error(f"U3V: connect failed: {e}")
            return False

    def _send_gvcp(self, command: int, address: int, length: int, value: int = 0) -> Optional[bytes]:
        req_id = self._req_id
        self._req_id += 1

        if command == READREG:
            packet = struct.pack("<BBHHHI", 0x42, 0x00, command, length, req_id, address)
        elif command in (WRITEREG, CONTROL):
            packet = struct.pack("<BBHHHII", 0x42, 0x00, command, length, req_id, address, value)
        else:
            return None

        try:
            self.dev.write(CTRL_OUT, packet, timeout=2000)
            resp = self.dev.read(CTRL_IN, 512, timeout=5000)
            return bytes(resp)
        except Exception as e:
            logger.debug(f"GVCP 0x{command:04X} addr=0x{address:08X} failed: {e}")
            return None

    def read_register(self, address: int) -> Optional[int]:
        resp = self._send_gvcp(READREG, address, 1)
        if resp is None or len(resp) < 16:
            return None
        try:
            cc, ptype, cmd, length, req_id, reserved, data = struct.unpack("<BBHHHII", resp[:16])
            if cmd == READREG and ptype == 0x00:
                return data
        except Exception:
            pass
        return None

    def write_register(self, address: int, value: int) -> bool:
        resp = self._send_gvcp(WRITEREG, address, 1, value)
        if resp is None or len(resp) < 12:
            return False
        try:
            cc, ptype, cmd, length, req_id, reserved = struct.unpack("<BBHHHI", resp[:12])
            return cmd == WRITEREG and ptype == 0x00
        except Exception:
            return False

    def capture(self) -> Optional[np.ndarray]:
        if not self.connected:
            return None
        try:
            data = self.dev.read(STREAM_IN, 1024 * 1024, timeout=5000)
            data = bytes(data)
            if len(data) < 64:
                return None

            magic, prefix, flags, payload_size, frame_id, ts, pix_fmt, w, h = struct.unpack(
                "<IHHIIIIII", data[:32]
            )
            if magic != 0x4C494F4E:
                return None

            img_data = data[64:64 + payload_size]
            n = len(img_data)
            if n == w * h * 3:
                img = np.frombuffer(img_data, dtype=np.uint8).reshape(h, w, 3)
                return img
            elif n == w * h:
                img = np.frombuffer(img_data, dtype=np.uint8).reshape(h, w)
                return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            return None
        except Exception as e:
            logger.error(f"U3V: capture failed: {e}")
            return None

    def disconnect(self):
        if self.dev is not None:
            try:
                usb.util.release_interface(self.dev, 0)
            except Exception:
                pass
            self.dev = None
        self.connected = False

    def get_camera_info(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "connected": self.connected,
            "resolution": self.settings["resolution"],
            "name": "Zeiss Axiocam 208 (U3V)",
            "model": "Axiocam 208",
        }

    def __repr__(self) -> str:
        return f"U3VCamera(index={self.camera_index}, connected={self.connected})"