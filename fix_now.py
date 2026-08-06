"""Kill Labscope, reset USB, test GenTL - all in one."""
import subprocess, time, os, sys

# 1. Kill Labscope
subprocess.run(["taskkill", "/F", "/IM", "Labscope.exe"], capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
subprocess.run(["taskkill", "/F", "/IM", "LabscopeService.exe"], capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
time.sleep(2)

# 2. Reset USB device
import usb.core
dev = usb.core.find(idVendor=0x0758, idProduct=0x6002)
if dev:
    print("Resetting USB...")
    dev.reset()
    time.sleep(5)
    print("USB reset done")

# 3. Test GenTL
from camera.zeiss_camera import ZeissCamera
cam = ZeissCamera(backend="gentl")
print("Connecting via GenTL...")
if cam.connect():
    print("SUCCESS! Connected!")
    frame = cam.capture()
    print(f"Frame: {frame.shape if frame is not None else 'None'}")
    cam.disconnect()
else:
    print("GenTL failed")

# 4. Test U3V direct
print("\nTrying U3V direct...")
from camera.u3v_camera import U3VCamera
cam2 = U3VCamera()
if cam2.connect():
    print("U3V SUCCESS!")
    cam2.disconnect()
else:
    print("U3V failed")