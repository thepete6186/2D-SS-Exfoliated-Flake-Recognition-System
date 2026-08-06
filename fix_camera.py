"""Kill Labscope, disable restart service, reset USB, test camera."""
import subprocess
import time
import os
import sys

def run(cmd, timeout=15):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          creationflags=subprocess.CREATE_NO_WINDOW)
        return r
    except Exception as e:
        return None

print("=== Step 1: Kill Labscope ===")
for p in ["Labscope.exe", "LabscopeService.exe"]:
    r = run(["taskkill", "/F", "/IM", p])
    print(f"  {p}: {'Killed' if r and r.returncode == 0 else 'Not running or failed'}")

time.sleep(1)

print("\n=== Step 2: Disable Labscope service via registry ===")
# Disable the service that restarts Labscope
r = run(["reg", "add", r"HKLM\SYSTEM\CurrentControlSet\Services\LabscopeFileTransferService",
         "/v", "Start", "/t", "REG_DWORD", "/d", "4", "/f"])
print(f"  Reg result: {r.stdout.strip() if r else 'failed'}")

print("\n=== Step 3: Reset USB device ===")
try:
    import usb.core
    dev = usb.core.find(idVendor=0x0758, idProduct=0x6002)
    if dev:
        try:
            dev.reset()
            print("  USB reset OK")
        except Exception as e:
            print(f"  USB reset failed: {e}")
        time.sleep(3)
except Exception as e:
    print(f"  pyusb error: {e}")

print("\n=== Step 4: Test GenTL ===")
sys.path.insert(0, '.')
try:
    from camera.zeiss_camera import ZeissCamera, _find_gentl_producers
    producers = _find_gentl_producers()
    print(f"  Producers: {producers}")
    
    cam = ZeissCamera(camera_index=0, backend="gentl")
    if cam.connect():
        print("  >>> SUCCESS! GenTL connected!")
        frame = cam.capture()
        if frame is not None:
            print(f"  Frame captured: {frame.shape}")
        else:
            print("  Frame capture failed")
        cam.disconnect()
    else:
        print("  GenTL failed")
except Exception as e:
    print(f"  GenTL error: {e}")

print("\n=== Step 5: Test MMCore ===")
try:
    from pymmcore_plus import CMMCorePlus
    mm = CMMCorePlus()
    mm_path = os.path.join(
        os.path.expanduser("~"),
        "AppData", "Local", "pymmcore-plus", "pymmcore-plus", "mm",
        "Micro-Manager_2.0.3_20260805"
    )
    if os.path.isdir(mm_path):
        mm.setDeviceAdapterSearchPaths([mm_path])
    
    axio = mm.getAvailableDevices("AxioCam")
    print(f"  AxioCam: {axio}")
    if axio:
        dev_name = axio[0]
        mm.loadDevice(dev_name, "AxioCam", dev_name)
        mm.initializeDevice(dev_name)
        mm.setCameraDevice(dev_name)
        w = mm.getImageWidth()
        h = mm.getImageHeight()
        print(f"  >>> SUCCESS! Resolution: {w}x{h}")
        mm.snapImage()
        img = mm.getImage()
        print(f"  Image: {img.shape if hasattr(img, 'shape') else type(img)}")
        mm.unloadAllDevices()
except Exception as e:
    print(f"  MMCore error: {e}")