# Camera Module

Camera interface for **Zeiss Axiocam 208 Color** scientific camera.

## Overview

This module provides camera capture functionality for the 2D material transfer system. It supports:

- **Micro-Manager / pymmcore** (recommended for Axiocam 208) - open-source SDK for scientific cameras
- **GenICam GenTL / Harvesters** - USB3 Vision camera control
- **OpenCV VideoCapture** (fallback) - basic capture if SDK not available

## Files

```
camera/
├── __init__.py           # Module exports
├── zeiss_camera.py       # GenTL/OpenCV camera class
├── mmcore_camera.py      # Micro-Manager (pymmcore) camera class
├── test_camera.py        # Test script for GenTL/OpenCV
├── test_mmcore_camera.py # Test script for Micro-Manager
├── setup_mmcore.py       # Micro-Manager setup helper
└── README.md             # This file
```

## Quick Start (Micro-Manager for Axiocam 208)

### 1. Install Dependencies

```bash
# Install pymmcore
pip install pymmcore pymmcore-plus

# Or install everything
pip install -r requirements.txt
```

### 2. Install Micro-Manager 2.0

1. Download Micro-Manager 2.0 from: https://micro-manager.org/
2. Install to the default location: `C:\Program Files\Micro-Manager-2.0`
3. Ensure the Zeiss Axiocam device adapter is in the `DeviceAdapters` folder
   - Common adapter: `Zeiss_Device_Adapter.dll` or similar
4. Set the `MICRO_MANAGER_PATH` environment variable:

```bash
setx MICRO_MANAGER_PATH "C:\Program Files\Micro-Manager-2.0"
```

Restart your terminal after setting the environment variable.

### 3. Run Setup Helper

```bash
# Check setup status, find adapter, list cameras
python camera/setup_mmcore.py

# Auto-fix MICRO_MANAGER_PATH environment variable
python camera/setup_mmcore.py --fix
```

### 4. Test Camera Connection

```bash
# From project root - auto-detect camera
python camera/test_mmcore_camera.py

# With a specific Micro-Manager config file
python camera/test_mmcore_camera.py --config "C:\Program Files\Micro-Manager-2.0\YourConfig.cfg"

# Just list available devices
python camera/test_mmcore_camera.py --list-devices
```

This will:
1. Initialize Micro-Manager Core
2. Find and connect to the Axiocam 208
3. Capture a test image
4. Save to `camera/test_mmcore_capture.png`

### 5. Use in Your Code

```python
from camera import MMCoreCamera

# Initialize camera
cam = MMCoreCamera(camera_index=0)
# Or with explicit config:
# cam = MMCoreCamera(mm_config_path=r"C:\Program Files\Micro-Manager-2.0\config.cfg")

# Connect
if cam.connect():
    # Capture image
    rgb = cam.capture()
    
    # Or save directly to file
    cam.capture_to_file("output.png")
    
    # Adjust settings
    cam.set_exposure(15000.0)  # 15us
    cam.set_gain(1.0)  # 1 dB
    
    # Get camera info
    info = cam.get_camera_info()
    print(info)
    
    # List available properties
    props = cam.get_available_properties()
    
    # Set/get arbitrary properties
    cam.set_property("Binning", "1")
    value = cam.get_property("PixelType")
    
    # Disconnect when done
    cam.disconnect()
```

## Camera Annotator (GUI)

The annotator now supports selecting between backends:

```bash
# Run annotator
python camera/camera_annotator.py
```

In the GUI:
1. Select backend from dropdown: `auto`, `opencv`, or `micro-manager`
2. Click "Connect Camera"
3. For `auto` mode: tries Micro-Manager first, then OpenCV/GenTL

## Camera Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `exposure_us` | float | 10000.0 | Exposure time in microseconds |
| `gain_db` | float | 0.0 | Gain in decibels |
| `white_balance` | str | 'auto' | White balance mode |
| `resolution` | tuple | (5472, 3648) | Image resolution (width, height) |

## API Reference

### `MMCoreCamera`

Camera driver using Micro-Manager Core (pymmcore) for scientific cameras like the Axiocam 208.

#### `__init__(camera_index=0, mm_config_path=None)`

- `camera_index`: Index of camera among detected devices
- `mm_config_path`: Optional path to Micro-Manager `.cfg` file

#### `connect() -> bool`

Connect to camera via Micro-Manager. Returns True on success.

#### `disconnect()`

Release camera resources.

#### `capture() -> np.ndarray or None`

Capture single image. Returns RGB numpy array (H, W, 3) or None on failure.

#### `capture_sequence(num_frames) -> np.ndarray or None`

Capture multiple frames. Returns array (N, H, W, 3).

#### `capture_to_file(output_path, format='png') -> bool`

Capture image and save to file. Returns True on success.

#### `set_exposure(exposure_us)`

Set exposure time in microseconds.

#### `set_gain(gain_db)`

Set gain in decibels.

#### `set_resolution(width, height)`

Set camera resolution.

#### `get_available_properties() -> list`

Get list of configurable camera properties.

#### `set_property(name, value) -> bool`

Set an arbitrary camera property.

#### `get_property(name) -> Any or None`

Get a camera property value.

#### `get_camera_info() -> dict`

Get camera information (name, model, resolution, backend).

### `ZeissCamera`

Camera driver using GenICam GenTL (via `harvesters`) or OpenCV fallback.

## Backend Details

### Micro-Manager (pymmcore) Backend - RECOMMENDED for Axiocam 208

**Requirements:**
- `pymmcore` and `pymmcore-plus` installed via pip
- Micro-Manager 2.0 installed (https://micro-manager.org/)
- Zeiss Axiocam device adapter in Micro-Manager's `DeviceAdapters` folder
- Camera connected via USB 3.0

**Features:**
- Full scientific camera control through Micro-Manager
- Access to all GenICam properties
- Proper Axiocam 208 support
- Exposure, gain, resolution, pixel format control

**Setup:**
1. Install Micro-Manager 2.0
2. Set `MICRO_MANAGER_PATH` environment variable
3. Run `python camera/setup_mmcore.py` to verify
4. Test with `python camera/test_mmcore_camera.py`

### GenTL/Harvesters Backend

**Requirements:**
- `pip install harvesters`
- Zeiss GenTL producer (`zeiss_u3vgentlk.cti` - from Zeiss ZEN/VisionSuite)
- Camera connected via USB 3.0

**Features:**
- Direct USB3 Vision access
- Full GenICam control

### OpenCV Backend

**Requirements:**
- OpenCV (`opencv-python`)
- Camera appears as standard webcam device (UVC)

**Limitations:**
- The Axiocam 208 is **NOT UVC-compatible** - OpenCV cannot open it directly
- Only fallback for other USB cameras

## Troubleshooting

### Camera not found via Micro-Manager

1. Check USB connection (use USB 3.0 port)
2. Verify camera powers on (LED indicator)
3. Make sure camera is not used by another app (Zeiss ZEN, Labscope, Micro-Manager GUI)
4. Check `MICRO_MANAGER_PATH` is set correctly:
   ```
   echo %MICRO_MANAGER_PATH%
   ```
5. Verify Zeiss adapter exists:
   ```
   dir "C:\Program Files\Micro-Manager-2.0\DeviceAdapters" | findstr -i "zeiss"
   ```
6. Run setup helper for diagnostics:
   ```
   python camera/setup_mmcore.py
   ```

### pymmcore import fails

```bash
pip install pymmcore pymmcore-plus
```

### Micro-Manager not installed

1. Download: https://micro-manager.org/
2. Install Micro-Manager 2.0
3. Restart terminal

### Camera used by another app

The Axiocam 208 can only be accessed by one app at a time. Close:
- Zeiss ZEN
- Zeiss Labscope
- Micro-Manager GUI
- Any other microscope software

### GenTL producer not found

Make sure Zeiss VisionSuite or ZEN is installed, which provides the GenTL producer (`zeiss_u3vgentlk.cti`).

### Low frame rate

- Use Micro-Manager backend instead of OpenCV
- Lower resolution (try 1920x1080 or 2736x1824)
- Increase exposure time
- Check USB 3.0 connection (not USB 2.0)

## Integration with Pipeline

To integrate with the flake detection pipeline:

```python
from camera import MMCoreCamera, ZeissCamera
from hsv_pipeline_semi.semi_supervised_pipeline import SemiSupervisedPipeline

# Initialize camera (try MMCore first, fallback to GenTL)
def connect_camera():
    mm_cam = MMCoreCamera()
    if mm_cam.connect():
        return mm_cam
    
    zeiss_cam = ZeissCamera()
    if zeiss_cam.connect():
        return zeiss_cam
    
    return None

# Connect
cam = connect_camera()
if cam is None:
    raise RuntimeError("No camera backend available")

# Capture image
rgb = cam.capture()

# Run pipeline (with pre-defined clicks or calibration)
result = pipeline.process(rgb, seed_points)

# Get flake locations
prob_map = result['probability_map']
# ... extract flake positions ...

# Disconnect
cam.disconnect()
```

## Notes

- The Axiocam 208 is a **scientific camera** with high resolution (up to 5472×3648)
- Micro-Manager (pymmcore) is the **recommended** backend since it provides proper
  support for scientific cameras via the open-source Micro-Manager SDK
- For consistent HSV detection, use **fixed exposure and gain** (don't use auto-exposure)
- The camera should be **fixed above the sample** (not on a stage) for consistent imaging