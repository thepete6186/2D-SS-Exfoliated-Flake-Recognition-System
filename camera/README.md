# Camera Module

Camera interface for **Zeiss XiCam 208 Color** scientific camera.

## Overview

This module provides camera capture functionality for the 2D material transfer system. It supports:

- **Zeiss Microscopy SDK** (preferred) - full camera control
- **OpenCV VideoCapture** (fallback) - basic capture if SDK not available

## Files

```
camera/
├── __init__.py           # Module exports
├── zeiss_camera.py       # Main camera class
├── test_camera.py        # Standalone test script
└── README.md             # This file
```

## Quick Start

### 1. Install Dependencies

```bash
# Install Zeiss Microscopy SDK (if available)
# Download from: https://www.zeiss.com/microscopy/en/downloads.html

# Or use OpenCV fallback (already installed)
pip install opencv-python numpy
```

### 2. Test Camera Connection

```bash
# From project root
python camera/test_camera.py
```

This will:
1. Try to connect via Zeiss SDK
2. Fall back to OpenCV if SDK not available
3. Capture a test image
4. Save to `camera/test_capture.png`

### 3. Use in Your Code

```python
from camera import ZeissCamera

# Initialize camera
cam = ZeissCamera(use_sdk=True)

# Connect
if cam.connect():
    # Capture image
    rgb = cam.capture()
    
    # Or save directly to file
    cam.capture_to_file("output.png")
    
    # Adjust settings
    cam.set_exposure(15000.0)  # 15ms
    cam.set_gain(1.0)  # 1 dB
    
    # Disconnect when done
    cam.disconnect()
```

## Camera Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `exposure_us` | float | 10000.0 | Exposure time in microseconds |
| `gain_db` | float | 0.0 | Gain in decibels |
| `white_balance` | str | 'auto' | White balance mode |
| `resolution` | tuple | (5472, 3648) | Image resolution (width, height) |

## API Reference

### `ZeissCamera`

#### `__init__(use_sdk=True, camera_index=0)`

- `use_sdk`: Try Zeiss SDK first, fallback to OpenCV
- `camera_index`: OpenCV camera index (usually 0 or 1)

#### `connect() -> bool`

Connect to camera. Returns True on success.

#### `disconnect()`

Release camera resources.

#### `capture() -> np.ndarray or None`

Capture single image. Returns RGB numpy array (H, W, 3) or None on failure.

#### `capture_to_file(output_path, format='png') -> bool`

Capture image and save to file. Returns True on success.

#### `set_exposure(exposure_us)`

Set exposure time in microseconds.

#### `set_gain(gain_db)`

Set gain in decibels.

#### `set_resolution(width, height)`

Set camera resolution (OpenCV only).

#### `get_settings() -> dict`

Get current camera settings.

#### `get_camera_info() -> dict`

Get camera information (name, model, resolution, backend).

## Backend Details

### Zeiss SDK Backend

**Requirements:**
- Zeiss Microscopy SDK installed
- Python bindings available (`zeiss.microscopy` or `pyzeiss`)

**Features:**
- Full camera control
- Hardware triggering (if supported)
- Advanced settings (white balance, gamma, etc.)

**Note:** The SDK API may vary by version. You may need to adjust the method names in `zeiss_camera.py` to match your SDK version.

### OpenCV Backend

**Requirements:**
- OpenCV (`opencv-python`)
- Camera appears as standard webcam device

**Features:**
- Basic capture
- Limited settings control (depends on camera driver)
- Works with any UVC-compatible camera

**Limitations:**
- Exposure/gain control may not work for all cameras
- No hardware triggering
- Lower performance than SDK

## Troubleshooting

### Camera not found

1. Check USB connection
2. Verify camera appears in Device Manager (Windows) or `lsusb` (Linux)
3. Try different camera indices (0, 1, 2...)
4. Install Zeiss SDK if using SDK mode

### SDK import fails

The Zeiss SDK module name varies by version. Common names:
- `zeiss.microscopy`
- `pyzeiss`
- `microscopy`

Check your SDK documentation for the correct import.

### Low frame rate

- Use SDK backend instead of OpenCV
- Lower resolution
- Reduce exposure time
- Check USB 3.0 connection (not USB 2.0)

### Exposure/gain not working

- OpenCV backend has limited camera control
- Use SDK backend for full control
- Some cameras require proprietary SDK for exposure control

## Integration with Pipeline

To integrate with the flake detection pipeline:

```python
from camera import ZeissCamera
from hsv_pipeline_semi.semi_supervised_pipeline import SemiSupervisedPipeline

# Initialize camera and pipeline
cam = ZeissCamera()
pipeline = SemiSupervisedPipeline()

# Connect camera
cam.connect()

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

## Next Steps

1. **Install Zeiss SDK** and test with `test_camera.py`
2. **Adjust SDK API calls** in `zeiss_camera.py` to match your SDK version
3. **Integrate with pipeline** for automated capture and detection
4. **Add GUI controls** for camera settings in transfer-stage-control

## Notes

- The XiCam 208 is a **scientific camera** with high resolution (up to 5472×3648)
- For consistent HSV detection, use **fixed exposure and gain** (don't use auto-exposure)
- Consider using **hardware triggering** for precise timing (requires SDK)
- The camera should be **fixed above the sample** (not on a stage) for consistent imaging