"""Check Zeiss services and reset camera USB."""
import subprocess, os, time

print("=== Zeiss Services ===")
try:
    r = subprocess.run(["sc","query","type=","service","state=","all"],
        capture_output=True, text=True, timeout=15,
        creationflags=subprocess.CREATE_NO_WINDOW)
    lines = r.stdout.split('\n')
    for i, line in enumerate(lines):
        if any(x in line.lower() for x in ['zeiss','axio','labscope','zen']):
            for j in range(max(0,i-1), min(len(lines),i+3)):
                print(f"  {lines[j].strip()}")
            print("  ---")
except Exception as e:
    print(f"  sc query failed: {e}")

print("\n=== Zeiss Processes ===")
try:
    r = subprocess.run(["tasklist","/fo","csv"],
        capture_output=True, text=True, timeout=10,
        creationflags=subprocess.CREATE_NO_WINDOW)
    for line in r.stdout.split('\n'):
        low = line.lower()
        if any(x in low for x in ['zeiss','labscope','zen','axio','mmstudio','micro-manager']):
            print(f"  {line.strip()}")
except Exception as e:
    print(f"  tasklist failed: {e}")

print("\n=== Reset USB device ===")
try:
    import usb.core, usb.util
    dev = usb.core.find(idVendor=0x0758, idProduct=0x6002)
    if dev:
        print(f"  Found: {dev.product}")
        try:
            dev.reset()
            print("  Reset OK")
            time.sleep(2)
        except Exception as e:
            print(f"  Reset failed: {e}")
        dev2 = usb.core.find(idVendor=0x0758, idProduct=0x6002)
        print(f"  After reset: {'present' if dev2 else 'GONE'}")
    else:
        print("  Not found")
except Exception as e:
    print(f"  pyusb error: {e}")

print("\n=== Test control transfer ===")
try:
    import usb.core, usb.util
    dev = usb.core.find(idVendor=0x0758, idProduct=0x6002)
    if dev:
        try:
            usb.util.claim_interface(dev, 0)
            print("  Interface claimed")
            try:
                desc = dev.ctrl_transfer(0x80, 0x06, 0x0100, 0, 18, timeout=2000)
                print(f"  Device desc: {bytes(desc).hex()}")
            except Exception as e:
                print(f"  Control transfer failed: {e}")
            usb.util.release_interface(dev, 0)
        except Exception as e:
            print(f"  Claim failed: {e}")
except Exception as e:
    print(f"  pyusb error: {e}")