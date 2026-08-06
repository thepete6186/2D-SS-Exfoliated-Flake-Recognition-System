"""Test direct U3V protocol communication with the Zeiss Axiocam 208."""
import usb.core
import usb.util
import struct
import sys
import time

ZEISS_VID = 0x0758
ZEISS_PID = 0x6002

print("=== Direct U3V Communication Test ===\n")

# Find device
dev = usb.core.find(idVendor=ZEISS_VID, idProduct=ZEISS_PID)
if dev is None:
    print("ERROR: Device not found")
    sys.exit(1)

print(f"Found: {dev.manufacturer} / {dev.product}")
print(f"Bus: {dev.bus}, Address: {dev.address}")

# Claim interface
print("\nClaiming interface 0...")
usb.util.claim_interface(dev, 0)
print("Interface claimed!")

# Find endpoints
ep_out = None
ep_in = None
for cfg in dev:
    for intf in cfg:
        for ep in intf:
            addr = ep.bEndpointAddress
            attrs = ep.bmAttributes
            print(f"  EP 0x{addr:02X} Type={attrs:02X} MaxPacket={ep.wMaxPacketSize}")
            if usb.util.endpoint_direction(addr) == usb.util.ENDPOINT_OUT:
                if ep_out is None:
                    ep_out = ep.bEndpointAddress
            else:
                if ep_in is None:
                    ep_in = ep.bEndpointAddress

print(f"\nControl EPs: OUT=0x{ep_out:02X} IN=0x{ep_in:02X}")

# U3V Control Protocol - SSCCP (SuperSpeed Camera Control Protocol)
# The U3V control channel uses a specific packet format:
#   Bytes 0-3: Magic "LION" (0x4C494F4E)
#   Bytes 4-5: Prefix (0x0000)
#   Bytes 6-7: Flags (0x0004 = REQUEST, 0x0006 = ACK)
#   Bytes 8-11: Command ID
#   Bytes 12-15: Length
#   Bytes 16-19: Request ID
#   Bytes 20-23: Target address
#   Bytes 24-31: Reserved

def send_u3v_command(dev, ep_out, ep_in, command, address, length, req_id, payload=b''):
    """Send a U3V command packet and read response."""
    packet = b''
    packet += struct.pack('<I', 0x4C494F4E)  # Magic "LION"
    packet += struct.pack('<H', 0x0000)       # Prefix
    packet += struct.pack('<H', 0x0004)       # Flags (REQUEST)
    packet += struct.pack('<I', command)      # Command
    packet += struct.pack('<I', length)       # Length
    packet += struct.pack('<I', req_id)       # Request ID
    packet += struct.pack('<I', address)      # Target address
    packet += struct.pack('<Q', 0)            # Reserved
    packet += payload
    
    print(f"  Sending cmd=0x{command:04X} addr=0x{address:08X} len={length} ({len(packet)} bytes)")
    try:
        sent = dev.write(ep_out, packet, timeout=2000)
        print(f"  Sent {sent} bytes")
    except Exception as e:
        print(f"  Write failed: {e}")
        return None
    
    try:
        resp = dev.read(ep_in, 1024, timeout=5000)
        print(f"  Response ({len(resp)} bytes): {bytes(resp).hex()}")
        return bytes(resp)
    except usb.core.USBTimeoutError:
        print("  TIMEOUT - no response!")
        return None
    except Exception as e:
        print(f"  Read error: {e}")
        return None

# Test 1: Read magic register (0x0000)
print("\n--- Test 1: READREG 0x0000 (Magic) ---")
resp = send_u3v_command(dev, ep_out, ep_in, 0x0080, 0x0000, 4, 1)

# Test 2: Read device model (0x0128)
print("\n--- Test 2: READREG 0x0128 (Model) ---")
resp = send_u3v_command(dev, ep_out, ep_in, 0x0080, 0x0128, 4, 2)

# Test 3: Try GVCP format (without LION magic)
print("\n--- Test 3: GVCP format READREG ---")
# GVCP header: CC(1) + PacketType(1) + Command(2) + Length(2) + ReqID(2) + Address(4)
packet = struct.pack('<BBHHHI',
    0x42,    # CC = 0x42 (Control Channel)
    0x00,    # Packet Type = 0x00 (CMD)
    0x0080,  # Command = READREG
    1,       # Length (in 4-byte words)
    1,       # Request ID
    0x0000   # Address
)
print(f"  Packet: {packet.hex()}")
try:
    sent = dev.write(ep_out, packet, timeout=2000)
    print(f"  Sent {sent} bytes")
    resp = dev.read(ep_in, 1024, timeout=5000)
    print(f"  Response: {bytes(resp).hex()}")
except Exception as e:
    print(f"  Error: {e}")

# Test 4: Try U3V discovery via control transfer
print("\n--- Test 4: U3V Discovery ---")
# U3V discovery uses standard USB control transfers
# Try to get the U3V camera control descriptor
for desc_type in [0x12, 0x11, 0x29, 0x24]:
    try:
        resp = dev.ctrl_transfer(
            bmRequestType=0x80,
            bRequest=0x06,
            wValue=(desc_type << 8) | 0,
            wIndex=0,
            data_or_wLength=64,
            timeout=2000
        )
        print(f"  Desc 0x{desc_type:02X}: {bytes(resp).hex()}")
    except Exception as e:
        print(f"  Desc 0x{desc_type:02X} failed: {type(e).__name__}")

# Test 5: Try to read the U3V control channel via bulk
print("\n--- Test 5: Read from control EP without sending ---")
try:
    resp = dev.read(ep_in, 1024, timeout=1000)
    print(f"  Got: {bytes(resp).hex()}")
except Exception as e:
    print(f"  No data: {type(e).__name__}")

# Release
print("\nReleasing interface...")
usb.util.release_interface(dev, 0)
print("Done!")