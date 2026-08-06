"""Reset USB device and test MMCore connection."""
import usb.core, usb.util, time, os, sys

print("=== Reset USB device ===")
dev = usb.core.find(idVendor=0x0758, idProduct=0x6002)
if dev:
    print(f"  Found: {dev.product}")
    try:
        dev.reset()
        print("  Reset OK")
        time.sleep(3)
    except Exception as e:
        print(f"  Reset failed: {e}")
else:
    print("  Not found")

print("\n=== Test MMCore connection ===")
try:
    from pymmcore_plus import CMMCorePlus
    mm = CMMCorePlus()
    
    mm_path = os.path.join(
        os.path.expanduser("~"),
        "AppData", "Local", "pymmcore-plus", "pymmcore-plus", "mm",
        "Micro-Manager_2.0.3_20260805"
    )
    if os.path.isdir(mm_path):
        print(f"  MM path: {mm_path}")
        mm.setDeviceAdapterSearchPaths([mm_path])
    
    # Find AxioCam devices
    axio_devs = mm.getAvailableDevices("AxioCam")
    print(f"  AxioCam devices: {axio_devs}")
    
    if axio_devs:
        dev_name = axio_devs[0]
        print(f"  Loading device: {dev_name}")
        mm.loadDevice(dev_name, "AxioCam", dev_name)
        mm.initializeDevice(dev_name)
        mm.setCameraDevice(dev_name)
        
        # Read camera info
        print(f"  Camera device: {mm.getCameraDevice()}")
        try:
            w = mm.getImageWidth()
            h = mm.getImageHeight()
            print(f"  Resolution: {w}x{h}")
        except Exception as e:
            print(f"  Resolution error: {e}")
        
        # Try to snap an image
        print("  Snapping image...")
        try:
            mm.snapImage()
            img = mm.getImage()
            print(f"  Image captured: {img.shape if hasattr(img, 'shape') else type(img)}")
        except Exception as e:
            print(f"  Snap failed: {e}")
        
        mm.unloadAllDevices()
        print("  SUCCESS! Camera connected via MMCore!")
    else:
        print("  No AxioCam devices found!")
except Exception as e:
    print(f"  MMCore error: {e}")