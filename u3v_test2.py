"""Test U3V with correct LION magic and reset camera."""
import usb.core, usb.util, struct, subprocess, time, os, sys

# Kill Labscope that holds the camera
subprocess.run(["taskkill", "/F", "/IM", "Labscope.exe"], capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
subprocess.run(["taskkill", "/F", "/IM", "LabscopeService.exe"], capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
time.sleep(1)

# Reset camera USB
dev = usb.core.find(idVendor=0x0758, idProduct=0x6002)
if dev:
    try:
        dev.reset()
        print("USB reset OK")
    except Exception as e:
        print(f"Reset failed: {e}")
    time.sleep(2)
    dev = usb.core.find(idVendor=0x0758, idProduct=0x6002)

if not dev:
    print("Camera not found!")
    sys.exit(1)

print(f"Found: {dev.product}")
usb.util.claim_interface(dev, 0)

# SSCCP READREG: CORRECT magic bytes = "LION" on wire
# struct.pack('<I', 0x4E4F494C) gives bytes 4C 49 4F 4E = "LION"
def ssccp_readreg(addr, reqid):
    pkt = struct.pack('<IHHIIIIQ', 0x4E4F494C, 0x0000, 0x0004, 0x0080, 4, reqid, addr, 0)
    assert len(pkt) == 32, f"Packet length {len(pkt)}"
    print(f"  READREG 0x{addr:08X} pkt={pkt.hex()}")
    try:
        dev.write(0x02, pkt, timeout=2000)
        resp = dev.read(0x82, 512, timeout=5000)
        print(f"  resp ({len(resp)}): {bytes(resp).hex()}")
        if len(resp) >= 32:
            magic, prefix, flags, cmd, length, reqid_r, addr_r, data = struct.unpack('<IHHIIIII', bytes(resp[:32]))
            print(f"  magic={bytes(resp[:4]).decode('ascii','replace')!r} prefix=0x{prefix:04X} flags=0x{flags:04X} "
                  f"cmd=0x{cmd:04X} len={length} reqid={reqid_r} addr=0x{addr_r:08X}")
            print(f"  DATA = 0x{data:08X}")
            return data
    except Exception as e:
        print(f"  FAILED: {e}")
    return None

print("\n=== Test SSCCP READREG ===")
v = ssccp_readreg(0x0000, 1)
if v is not None:
    # Try more registers if first worked
    print("\n=== More registers ===")
    for addr in [0x0004, 0x0008, 0x0128, 0x0800]:
        ssccp_readreg(addr, 2)

usb.util.release_interface(dev, 0)
print("\nDone.")