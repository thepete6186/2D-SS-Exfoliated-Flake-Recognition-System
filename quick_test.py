"""Kill Labscope and immediately test camera connection."""
import subprocess, time, os, sys

# Kill Labscope
print("Killing Labscope...")
subprocess.run(["taskkill", "/F", "/IM", "Labscope.exe"], 
    capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
subprocess.run(["taskkill", "/F", "/IM", "LabscopeService.exe"], 
    capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
time.sleep(1)

# Reset USB device
print("Resetting USB device...")
try:
    import usb.core
    dev = usb.core.find(idVendor=0x0758, idProduct=0x6002)
    if dev:
        dev.reset()
        print("  USB reset OK")
        time.sleep(2)
except Exception as e:
    print(f"  USB reset failed: {e}")

# Immediately test MMCore
print("\nTesting MMCore connection...")
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
    
    axio_devs = mm.getAvailableDevices("AxioCam")
    print(f"  AxioCam devices: {axio_devs}")
    
    if axio_devs:
        dev_name = axio_devs[0]
        print(f"  Loading: {dev_name}")
        mm.loadDevice(dev_name, "AxioCam", dev_name)
        mm.initializeDevice(dev_name)
        mm.setCameraDevice(dev_name)
        
        w = mm.getImageWidth()
        h = mm.getImageHeight()
        print(f"  Resolution: {w}x{h}")
        
        print("  Snapping image...")
        mm.snapImage()
        img = mm.getImage()
        print(f"  Image: {img.shape if hasattr(img, 'shape') else type(img)}")
        
        mm.unloadAllDevices()
        print("  SUCCESS! Camera connected!")
    else:
        print("  No AxioCam devices!")
except Exception as e:
    print(f"  MMCore error: {e}")

# Also test GenTL
print("\nTesting GenTL connection...")
try:
    sys.path.insert(0, '.')
    from camera.zeiss_camera import ZeissCamera
    cam = ZeissCamera(camera_index=0, backend="gentl")
    if cam.connect():
        print("  SUCCESS! GenTL connected!")
        frame = cam.capture()
        if frame is not None:
            print(f"  Frame: {frame.shape}")
        cam.disconnect()
    else:
        print("  GenTL failed")
except Exception as e:
    print(f"  GenTL error: {e}")