"""Test Usb3CamHS adapter (correct for USB3 Vision cameras)."""
import os, sys, subprocess, time

# Kill Labscope
subprocess.run(["taskkill", "/F", "/IM", "Labscope.exe"], capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
subprocess.run(["taskkill", "/F", "/IM", "LabscopeService.exe"], capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
time.sleep(1)

# Reset camera
try:
    import usb.core
    dev = usb.core.find(idVendor=0x0758, idProduct=0x6002)
    if dev:
        dev.reset()
        print("USB reset OK")
        time.sleep(3)
except Exception as e:
    print(f"USB reset: {e}")

# Ensure Zeiss GenTL DLLs are on PATH for this process
zeiss_path = r"C:\Program Files\Zeiss\ZeissVisionSuite_Compact\zeiss_u3vgentl\64"
if os.path.isdir(zeiss_path) and zeiss_path not in os.environ.get("PATH", ""):
    os.environ["PATH"] = zeiss_path + os.pathsep + os.environ.get("PATH", "")
    print(f"Added to PATH: {zeiss_path}")

print(f"PATH contains Zeiss: {'zeiss' in os.environ.get('PATH', '').lower()}")

from pymmcore_plus import CMMCorePlus
mm = CMMCorePlus()

mm_path = os.path.join(
    os.path.expanduser("~"),
    "AppData", "Local", "pymmcore-plus", "pymmcore-plus", "mm",
    "Micro-Manager_2.0.3_20260805"
)
print(f"MM path: {mm_path}")
mm.setDeviceAdapterSearchPaths([mm_path])

# Try Usb3CamHS adapter
print("\n=== Usb3CamHS adapter ===")
try:
    devs = mm.getAvailableDevices("Usb3CamHS")
    print(f"  Devices: {devs}")
    if devs:
        dev_name = devs[0]
        print(f"  Loading: {dev_name}")
        mm.loadDevice(dev_name, "Usb3CamHS", dev_name)
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
    print(f"  Usb3CamHS failed: {e}")

# Also try AxioCam with the GenTL DLLs on PATH
print("\n=== AxioCam adapter (with Zeiss DLLs on PATH) ===")
try:
    devs = mm.getAvailableDevices("AxioCam")
    print(f"  Devices: {devs}")
    if devs:
        dev_name = devs[0]
        print(f"  Loading: {dev_name}")
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
    print(f"  AxioCam failed: {e}")

print("\nDone.")