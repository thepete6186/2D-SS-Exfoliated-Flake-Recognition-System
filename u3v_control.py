"""Test U3V control on different endpoint pairs."""
import usb.core
import usb.util
import struct
import sys

ZEISS_VID = 0x0758
ZEISS_PID = 0x6002

dev = usb.core.find(idVendor=ZEISS_VID, idProduct=ZEISS_PID)
if dev is None:
    print("Device not found")
    sys.exit(1)

print(f"Found: {dev.product}")
usb.util.claim_interface(dev, 0)

# List all endpoints
eps = []
for cfg in dev:
    for intf in cfg:
        for ep in intf:
            eps.append(ep)
            print(f"  EP 0x{ep.bEndpointAddress:02X} Dir={'IN' if ep.bEndpointAddress & 0x80 else 'OUT'} Type={ep.bmAttributes}")

# Try GVCP READREG on each OUT endpoint, read from corresponding IN
def gvcp_readreg(dev, ep_out_addr, ep_in_addr, address, req_id):
    """GVCP READREG packet"""
    packet = struct.pack('<BBHHHI',
        0x42,    # CC
        0x00,    # Packet type (CMD)
        0x0080,  # READREG
        1,       # Length (1 word)
        req_id,  # Request ID
        address  # Target address
    )
    print(f"  -> EP 0x{ep_out_addr:02X}: {packet.hex()}")
    try:
        sent = dev.write(ep_out_addr, packet, timeout=2000)
        print(f"     Sent {sent} bytes")
    except Exception as e:
        print(f"     Write failed: {e}")
        return None
    
    try:
        resp = dev.read(ep_in_addr, 1024, timeout=3000)
        print(f"  <- EP 0x{ep_in_addr:02X}: {bytes(resp).hex()[:100]}")
        return bytes(resp)
    except Exception as e:
        print(f"     Read failed: {e}")
        return None

# Try all OUT/IN endpoint pairs
print("\n=== Test 1: READREG on EP 0x01/0x81 (first pair) ===")
gvcp_readreg(dev, 0x01, 0x81, 0x0000, 1)

print("\n=== Test 2: READREG on EP 0x02/0x82 (second pair) ===")
gvcp_readreg(dev, 0x02, 0x82, 0x0000, 1)

print("\n=== Test 3: U3V LION format on EP 0x02/0x82 ===")
packet = struct.pack('<IHHIIIIQ',
    0x4C494F4E, 0x0000, 0x0004, 0x0080, 4, 1, 0x0000, 0
)
try:
    sent = dev.write(0x02, packet, timeout=2000)
    print(f"  Sent {sent} bytes")
    resp = dev.read(0x82, 1024, timeout=3000)
    print(f"  Response: {bytes(resp).hex()[:200]}")
except Exception as e:
    print(f"  Failed: {e}")

print("\n=== Test 4: Try to read stream from EP 0x82 ===")
try:
    resp = dev.read(0x82, 1024, timeout=2000)
    print(f"  Got {len(resp)} bytes from EP 0x82")
except Exception as e:
    print(f"  No data: {type(e).__name__}")

print("\n=== Test 5: Full U3V discovery ===")
# U3V discovery uses control transfers with specific bRequests
# USB Device Capability Descriptor with bDevCapabilityType = 0x13 (U3V)
for desc_type in [0x13, 0x14, 0x15]:
    try:
        resp = dev.ctrl_transfer(
            bmRequestType=0x80,
            bRequest=0x06,
            wValue=(desc_type << 8) | 0,
            wIndex=0,
            data_or_wLength=32,
            timeout=2000
        )
        print(f"  Desc 0x{desc_type:02X}: {bytes(resp).hex()}")
    except Exception as e:
        print(f"  Desc 0x{desc_type:02X} failed: {type(e).__name__}")

usb.util.release_interface(dev, 0)
print("\nDone!")