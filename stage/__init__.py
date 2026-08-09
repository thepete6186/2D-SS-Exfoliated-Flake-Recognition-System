"""Stage control package for the Zolix ZC300 motion controller.

Public surface:

- Stage            — abstract interface (depend on this, not a driver)
- ZolixZC300       — Modbus RTU driver for the real controller
- SimulatedStage   — hardware-free double for tests and dry runs
- Stage*Error      — typed exception hierarchy

Smoke-test CLI for the lab PC: `python -m stage.zc300_smoke --help`.
"""

from stage.base import (
    AXES,
    ModbusError,
    Stage,
    StageAlarmError,
    StageBusyError,
    StageError,
    StageEstopError,
    StageLimitError,
    StageNotConnectedError,
    StageNotEnabledError,
    StageTimeoutError,
)
from stage.simulated import SimulatedStage
from stage.zolix_zc300 import ZolixZC300

__all__ = [
    "AXES",
    "ModbusError",
    "SimulatedStage",
    "Stage",
    "StageAlarmError",
    "StageBusyError",
    "StageError",
    "StageEstopError",
    "StageLimitError",
    "StageNotConnectedError",
    "StageNotEnabledError",
    "StageTimeoutError",
    "ZolixZC300",
]
