"""Test GenTL with only the correct architecture CTI file."""
import os
import sys
import struct

print(f"Python: {sys.version}")
print(f"Architecture: {'64-bit' if struct.calcsize('P') == 8 else '32-bit'}")

# Test the fixed _find_gentl_producers
sys.path.insert(0, '.')
from camera.zeiss_camera import _find_gentl_producers, ZeissCamera

print("\n=== GenTL Producer Discovery (fixed) ===")
producers = _find_gentl_producers()
print(f"Found {len(producers)} producer(s):")
for p in producers:
    print(f"  {p}")

print("\n=== Testing ZeissCamera GenTL connection ===")
cam = ZeissCamera(camera_index=0, backend="gentl")
if cam.connect():
    print("SUCCESS! Camera connected via GenTL!")
    info = cam.get_camera_info()
    print(f"  Info: {info}")
    # Try to capture a frame
    frame = cam.capture()
    if frame is not None:
        print(f"  Captured frame: {frame.shape}")
    else:
        print("  Failed to capture frame")
    cam.disconnect()
else:
    print("FAILED to connect via GenTL")
    print("  (Camera may be held by another app or not powered)")