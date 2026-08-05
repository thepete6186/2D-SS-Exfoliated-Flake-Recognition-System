#!/usr/bin/env python3
"""
Test script for Zeiss XiCam 208 camera.

Usage:
    python camera/test_camera.py
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from camera.zeiss_camera import ZeissCamera, CameraError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def main():
    print("=" * 60)
    print("Zeiss XiCam 208 Color - Camera Test")
    print("=" * 60)
    
    # Initialize camera (try SDK first, fallback to OpenCV)
    print("\n[1/4] Initializing camera...")
    cam = ZeissCamera(use_sdk=True)
    
    # Connect
    print("\n[2/4] Connecting to camera...")
    if not cam.connect():
        print("❌ FAILED: Could not connect to camera")
        print("\nTroubleshooting:")
        print("  1. Check camera is connected via USB")
        print("  2. Install Zeiss Microscopy SDK (if using SDK mode)")
        print("  3. Check camera appears in Device Manager (Windows)")
        return 1
    
    print(f"✓ Connected successfully")
    print(f"  Backend: {cam.backend}")
    
    # Print camera info
    info = cam.get_camera_info()
    print(f"  Camera: {info.get('name', 'Unknown')}")
    print(f"  Model: {info.get('model', 'Unknown')}")
    print(f"  Resolution: {info['resolution']}")
    
    # Capture test image
    print("\n[3/4] Capturing test image...")
    rgb = cam.capture()
    if rgb is None:
        print("❌ FAILED: Could not capture image")
        cam.disconnect()
        return 1
    
    print(f"✓ Captured successfully")
    print(f"  Shape: {rgb.shape}")
    print(f"  Dtype: {rgb.dtype}")
    print(f"  Min/Max: {rgb.min()} / {rgb.max()}")
    
    # Save test image
    output_dir = Path(__file__).parent
    output_path = output_dir / "test_capture.png"
    
    print(f"\n[4/4] Saving test image...")
    if cam.capture_to_file(output_path, format="png"):
        print(f"✓ Saved to: {output_path}")
    else:
        print("❌ FAILED: Could not save image")
        cam.disconnect()
        return 1
    
    # Test settings
    print("\n[Bonus] Testing camera settings...")
    try:
        cam.set_exposure(15000.0)  # 15ms
        print("  ✓ Exposure set to 15000 µs")
    except Exception as e:
        print(f"  ⚠ Exposure setting failed: {e}")
    
    try:
        cam.set_gain(1.0)  # 1 dB
        print("  ✓ Gain set to 1.0 dB")
    except Exception as e:
        print(f"  ⚠ Gain setting failed: {e}")
    
    # Disconnect
    cam.disconnect()
    
    print("\n" + "=" * 60)
    print("✓ Camera test completed successfully!")
    print("=" * 60)
    print(f"\nTest image saved to: {output_path}")
    print("You can open this file to verify the camera is working.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())