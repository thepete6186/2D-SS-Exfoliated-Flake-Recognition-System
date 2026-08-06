"""Test Usb3CamHS adapter with Zeiss DLLs on PATH."""
import os, sys

# Add Zeiss GenTL DLLs to PATH
zeiss_path = r"C:\Program Files\Zeiss\ZeissVisionSuite_Compact\zeiss_u3vgentl\64"
if os.path.isdir(zeiss_path):
    os.environ["PATH"] = zeiss_path + os.pathsep + os.environ.get("PATH", "")
    print(f"Added to PATH: {zeiss_path}")

from pymmcore_plus import CMMCorePlus
mm = CMMCorePlus()

mm_path = os.path.join(
    os.path.expanduser("~"),
    "AppData", "Local", "pymmcore-plus", "pymmcore-plus", "mm",
    "Micro-Manager_2.0.3_20260805"
)
print(f"MM path: {mm_path}")
mm.setDeviceAdapterSearchPaths([mm_path])

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
    import traceback
    traceback.print_exc()

print("\nDone.")