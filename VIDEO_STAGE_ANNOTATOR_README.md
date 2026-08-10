# Video Stage Annotator

Integrated camera annotation + Zolix ZC300 stage control application.

## Features

- **Live SmartCam video feed** - Real-time camera view from Zeiss Axiocam 208 (SmartCamApi backend only)
- **Point annotation with HSV** - Left-click to place points, HSV values recorded automatically
- **Substrate calibration** - Auto-detect or click-to-calibrate substrate HSV
- **Zolix ZC300 stage control** - Full X/Y/R axis motion control interface
- **Save/Load points** - Export/import point data with HSV to JSON

## Requirements

- Python 3.8+
- SmartCamApi.dll (comes with Zeiss Labscope)
- libusb0.dll driver for camera
- Zolix ZC300 stage (or use SimulatedStage for testing)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic usage (with simulated stage for testing):
```bash
python video_stage_annotator.py --simulate
```

### With real Zolix ZC300 stage:
```bash
python video_stage_annotator.py --port COM3
```

### Command-line options:
- `--port COM3` - Serial port for Zolix ZC300 (default: COM3)
- `--simulate` - Use SimulatedStage instead of real hardware

## Interface

### Left Panel: Video Feed
- **Left-click** - Add point (records HSV at click location)
- **Shift-click** - Add substrate sample (for calibration)
- **Right-click** - Remove last point
- **Mouse wheel** - Zoom in/out

### Right Panel: Controls

#### Stage Control (Zolix ZC300)
- **Port** - Serial port (e.g., COM3)
- **Simulate** - Checkbox for simulated stage
- **Connect Stage** - Connect/disconnect stage
- **Motion controls** - Per-axis relative/absolute moves
- **Home** - Home individual axes or all axes
- **Stop All** - Emergency stop
- **Speed** - Set movement speed (pulses/s)
- **Status** - Live position and motion state readout

#### Camera (SmartCam)
- **Connect Camera** - Connect to Zeiss Axiocam 208 via SmartCamApi
- Camera only uses SmartCam backend (no GenTL, OpenCV, or Micro-Manager)

#### Substrate HSV
- **Auto-Detect** - Automatically detect substrate HSV from current frame
- **Calibrate** - Toggle click mode for manual substrate sampling
- Displays current substrate HSV values

#### Points Management
- **Save Points** - Export points with HSV to JSON
- **Load Points** - Import points from JSON
- **Clear All** - Remove all points

## Keyboard Shortcuts

- **Left-click** - Add flake point
- **Shift+Left-click** - Add substrate sample
- **Right-click** - Remove last point
- **Mouse wheel** - Zoom

## Notes

- Camera uses SmartCamApi.dll only (native Zeiss SDK)
- Stage uses MODBUS RTU over USB virtual COM port
- All motion operations run in background threads to keep GUI responsive
- HSV values are recorded at full resolution (not display resolution)
- Points are stored in original image coordinates

## Troubleshooting

### Camera won't connect:
1. Ensure SmartCamApi.dll is installed (comes with Labscope)
2. Ensure libusb0 driver is installed for the camera
3. Make sure camera is not held by another app (Labscope, ZEN)
4. Try unplugging and replugging the camera

### Stage won't connect:
1. Check serial port (default COM3, adjust with --port)
2. Ensure stage is powered on
3. Check USB cable connection
4. Try SimulatedStage mode (--simulate) to test UI

## Files

- `video_stage_annotator.py` - Main application
- `camera/smartcam_camera.py` - SmartCamApi camera driver
- `stage/zolix_zc300.py` - Zolix ZC300 stage driver
- `stage/simulated.py` - Simulated stage for testing
- `hsv-pipeline-semi/semi_supervised_pipeline.py` - HSV extraction utilities

## Comparison with Original Tools

### vs. camera_annotator.py
- **Removed**: Load Image from File, Watch Labscope Folder, backend selection (GenTL/OpenCV/MMCore)
- **Added**: Zolix stage control interface
- **Camera**: SmartCam only (no other backends)

### vs. stage_control.py
- **Removed**: Standalone stage-only interface
- **Added**: Integrated video feed and point annotation
- **Combined**: Both tools in one unified interface

## Example Workflow

1. Launch: `python video_stage_annotator.py --simulate`
2. Connect camera (SmartCam button)
3. Connect stage (or use simulation)
4. Auto-detect or calibrate substrate HSV
5. Click on flakes to record points with HSV
6. Move stage to new position
7. Continue clicking flakes
8. Save points to JSON for later analysis