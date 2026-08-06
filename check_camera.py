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
print(f"bcdUSB: {hex(dev.bcdUSB)}")

# Detach kernel driver if needed
for interface in range(dev.bNumInterfaces):
    try:
        if dev.is_kernel_driver_active(interface):
            print(f"Detaching kernel driver from interface {interface}...")
            dev.detach_kernel_driver(interface)
    except Exception as e:
        print(f"  (interface {interface} kernel driver: {e})")

# Claim interface
print("\nClaiming interface 0...")
usb.util.claim_interface(dev, 0)
print("Interface claimed!")

# Find endpoints from the descriptor
ep_out = None
ep_in = None
ep_stream_out = None
ep_stream_in = None
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
                    ep_stream_out = ep.bEndpointAddress
            else:
                if ep_in is None:
                    ep_in = ep.bEndpointAddress
                else:
                    ep_stream_in = ep.bEndpointAddress

print(f"\nControl EPs: OUT=0x{ep_out:02X} IN=0x{ep_in:02X}")
print(f"Stream EPs: OUT=0x{ep_stream_out:02X} IN=0x{ep_stream_in:02X}")

# --- U3V Control Protocol ---
# The U3V control channel uses the SSCCP (SuperSpeed Camera Control Protocol)
# packet format which wraps GVCP commands over USB bulk transfers.
#
# Header format (32 bytes):
#   0-3:  Magic number (0x4C494F4E = "LION")
#   4-5:  Prefix (0x0000)
#   6-7:  Flags (0x0004 = request, 0x0006 = ack)
#   8-11: Command ID 
#   12-15: Length (payload length in bytes)
#   16-19: Request ID (host-generated)
#   20-23: Target address (register address for READREG/WRITEREG)
#   24-31: Reserved

def u3v_readreg(dev, ep_out, ep_in, address, num_regs=1, req_id=0x00000001):
    """Send a U3V READREG command and read the response."""
    # READREG command: 0x0080
    # For U3V, the packet structure is different from GVCP - it uses
    # a leading "U3V" check and different field layout
    
    # U3V control packet format (SSCCP):
    #   Bytes 0-1: Prefix = 0x0001
    #   Bytes 2-3: Flags = 0x0004 (REQUEST)
    #   Bytes 4-7: Command = 0x0080 (READREG) | (0x0000 << 16)
    #   Bytes 8-11: Length = num_regs * 4
    #   Bytes 12-15: Request ID
    #   Bytes 16-19: Target address
    #   Bytes 20-31: Reserved
    
    packet = b''
    packet += struct.pack('<I', 0x4C494F4E)  # Magic "LION"
    packet += struct.pack('<H', 0x0000)       # Prefix
    packet += struct.pack('<H', 0x0004)       # Flags (REQUEST)
    packet += struct.pack('<I', 0x0080)       # Command (READREG)
    packet += struct.pack('<I', num_regs * 4) # Length
    packet += struct.pack('<I', req_id)       # Request ID
    packet += struct.pack('<I', address)      # Target address
    packet += struct.pack('<Q', 0)            # Reserved (8 bytes)
    
    print(f"\nSending READREG for address 0x{address:08X} ({num_regs} regs)...")
    print(f"  Packet ({len(packet)} bytes): {packet.hex()}")
    
    sent = dev.write(ep_out, packet, timeout=2000)
    print(f"  Sent {sent} bytes")
    
    try:
        resp = dev.read(ep_in, 512, timeout=5000)
        print(f"  Response ({len(resp)} bytes): {bytes(resp).hex()}")
        return bytes(resp)
    except usb.core.USBTimeoutError:
        print("  TIMEOUT - no response from camera!")
        return None
    except Exception as e:
        print(f"  Error reading response: {e}")
        return None

def u3v_writereg(dev, ep_out, ep_in, address, value, req_id=0x00000002):
    """Send a U3V WRITEREG command and read the response."""
    # WRITEREG command: 0x0082
    packet = b''
    packet += struct.pack('<I', 0x4C494F4E)  # Magic "LION"
    packet += struct.pack('<H', 0x0000)       # Prefix
    packet += struct.pack('<H', 0x0004)       # Flags (REQUEST)
    packet += struct.pack('<I', 0x0082)       # Command (WRITEREG)
    packet += struct.pack('<I', 4)            # Length (1 register = 4 bytes)
    packet += struct.pack('<I', req_id)       # Request ID
    packet += struct.pack('<I', address)      # Target address
    packet += struct.pack('<I', value)        # Value to write
    packet += struct.pack('<I', 0)            # Reserved
    
    print(f"\nWRITEREG 0x{address:08X} = 0x{value:08X}...")
    sent = dev.write(ep_out, packet, timeout=2000)
    print(f"  Sent {sent} bytes")
    
    try:
        resp = dev.read(ep_in, 512, timeout=5000)
        print(f"  Response: {bytes(resp).hex()}")
        return bytes(resp)
    except Exception as e:
        print(f"  Error: {e}")
        return None

# Test 1: Read magic number register at 0x0000
print("\n--- Test 1: Read Magic Register (0x0000) ---")
resp1 = u3v_readreg(dev, ep_out, ep_in, 0x0000, num_regs=1, req_id=1)

# Test 2: Read camera model register
print("\n--- Test 2: Read Model Register (0x0128) ---")
resp2 = u3v_readreg(dev, ep_out, ep_in, 0x0128, num_regs=1, req_id=2)

# Test 3: Try different control packet format (without LION magic - some cameras use GVCP format directly)
print("\n--- Test 3: GVCP format READREG (no magic) at 0x0000 ---")
# GVCP format: CC=0x42 (control channel), command, length, req_id, address
packet = struct.pack('<BBHHII',
    0x42,          # CC (Control Channel) = 0x42
    0x00,          # Packet type = 0x00
    0x0080,        # Command - actually this is wrong, let me fix
    2,             # Length
    1,             # Request ID
    0              # Address
)
# Actually try the proper GVCP header format
packet = struct.pack('<HHHIIII',
    0x0052,  # Prefix (0x0000 + heartbeat flag?), 
    0x0000,  # Flags
    0x0080,  # Command (READREG)
    1,       # Length (words)
    1,       # Request ID
    0x0000,  # Address
    0        # Padding
)
sent = dev.write(ep_out, packet, timeout=2000)
print(f"  Sent {sent} bytes")
try:
    resp = dev.read(ep_in, 512, timeout=3000)
    print(f"  Response: {bytes(resp).hex()}")
except Exception as e:
    print(f"  Error: {e}")

# Test 4: Try to send a USB standard control request
print("\n--- Test 4: Standard USB Control Transfer ---")
try:
    # Get device descriptor (standard request)
    ddesc = dev.ctrl_transfer(
        bmRequestType=0x80,  # Device-to-host, Standard, Device
        bRequest=0x06,       # GET_DESCRIPTOR
        wValue=0x0100,       # Device descriptor
        wIndex=0,
        data_or_wLength=18,
        timeout=2000
    )
    print(f"  Device Descriptor ({len(ddesc)} bytes): {bytes(ddesc).hex()}")
    # Parse: bLength, bDescriptorType, bcdUSB, bDeviceClass, bDeviceSubClass, bDeviceProtocol, bMaxPacketSize0
    bLength, bDType, bcdUSB, bClass, bSubclass, bProtocol, bMaxPkt = struct.unpack('<BBHBBBB', bytes(ddesc[:8]))
    print(f"  bcdUSB=0x{bcdUSB:04X}, Class=0x{bClass:02X}, SubClass=0x{bSubclass:02X}, Protocol=0x{bProtocol:02X}")
except Exception as e:
    print(f"  Error: {e}")

# Test 5: Try to read camera description via READMEM
# First find the control channel via U3V discovery
print("\n--- Test 5: U3V Discovery Control ---")
# The U3V discovery response should be on EP IN
# Send a 0x0001 (GETDET) command? Actually for U3V, the "U3V discovery" 
# is done via standard control transfers. Let me try the U3V specific descriptor
try:
    # U3V camera control descriptor (U3V_CC_DESCRIPTOR = 0x12?)
    for desc_type in [0x12, 0x11, 0x29]:
        try:
            resp = dev.ctrl_transfer(
                bmRequestType=0x80,
                bRequest=0x06,
                wValue=(desc_type << 8) | 0,
                wIndex=0,
                data_or_wLength=64,
                timeout=2000
            )
            print(f"  Descriptor type 0x{desc_type:02X}: {bytes(resp).hex()}")
        except Exception as e:
            print(f"  Descriptor 0x{desc_type:02X} failed: {type(e).__name__}")
except Exception as e:
    print(f"  Error: {e}")

# Release
print("\nReleasing interface...")
usb.util.release_interface(dev, 0)
print("Done!")