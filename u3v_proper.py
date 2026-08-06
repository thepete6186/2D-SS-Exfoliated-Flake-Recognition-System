"""Proper U3V SSCCP control protocol test for Zeiss Axiocam 208."""
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

# U3V Control Channel: EP 0x02 (OUT) / EP 0x82 (IN)
# U3V Stream Channel: EP 0x01 (OUT) / EP 0x81 (IN)
CTRL_OUT = 0x02
CTRL_IN = 0x82

def u3v_readreg(dev, ep_out, ep_in, address, num_regs=1, req_id=1):
    """
    Send U3V READREG command using SSCCP format.
    
    SSCCP (SuperSpeed Camera Control Protocol) header:
    - [0:4]  Magic: "LION" (0x4C494F4E)
    - [4:6]  Prefix: 0x0002 (USB3 Vision)
    - [6:8]  Flags: 0x0004 (REQUEST)
    - [8:12] Command: 0x0080 (READREG)
    - [12:16] Length: num_regs * 4
    - [16:20] Request ID
    - [20:24] Target address
    - [24:28] Reserved (0)
    """
    packet = struct.pack('<IHHIIII',
        0x4C494F4E,  # Magic "LION"
        0x0002,      # Prefix (USB3 Vision)
        0x0004,      # Flags (REQUEST)
        0x0080,      # Command (READREG)
        num_regs * 4,  # Length in bytes
        req_id,      # Request ID
        address      # Target address
    )
    # SSCCP is 32 bytes total
    packet += struct.pack('<I', 0)  # Reserved
    
    print(f"  Sending READREG 0x{address:08X} (32 bytes): {packet.hex()}")
    try:
        sent = dev.write(ep_out, packet, timeout=2000)
        print(f"  Sent {sent} bytes")
    except Exception as e:
        print(f"  Write failed: {e}")
        return None
    
    try:
        resp = dev.read(ep_in, 512, timeout=5000)
        print(f"  Response ({len(resp)} bytes): {bytes(resp).hex()}")
        return bytes(resp)
    except Exception as e:
        print(f"  Read failed: {e}")
        return None

def u3v_writereg(dev, ep_out, ep_in, address, value, req_id=2):
    """
    Send U3V WRITEREG command.
    """
    packet = struct.pack('<IHHIIIII',
        0x4C494F4E,  # Magic "LION"
        0x0002,      # Prefix (USB3 Vision)
        0x0004,      # Flags (REQUEST)
        0x0082,      # Command (WRITEREG)
        4,           # Length (4 bytes = 1 register)
        req_id,      # Request ID
        address,     # Target address
        value        # Value to write
    )
    print(f"  Sending WRITEREG 0x{address:08X} = 0x{value:08X}")
    try:
        sent = dev.write(ep_out, packet, timeout=2000)
        print(f"  Sent {sent} bytes")
    except Exception as e:
        print(f"  Write failed: {e}")
        return None
    
    try:
        resp = dev.read(ep_in, 512, timeout=5000)
        print(f"  Response ({len(resp)} bytes): {bytes(resp).hex()}")
        return bytes(resp)
    except Exception as e:
        print(f"  Read failed: {e}")
        return None

def parse_readreg_response(resp):
    """Parse U3V READREG ACK response."""
    if resp is None or len(resp) < 32:
        return None
    magic, prefix, flags, cmd, length, req_id, addr, data = struct.unpack('<IHHIIIII', resp[:32])
    print(f"  Parsed: magic=0x{magic:08X} prefix=0x{prefix:04X} flags=0x{flags:04X} "
          f"cmd=0x{cmd:04X} len={length} req_id={req_id} addr=0x{addr:08X} data=0x{data:08X}")
    
    # Check if magic is "NOIL" (response magic)
    if magic == 0x4E4F494C or magic == 0x4C494F4E:
        return data
    return None

# Test 1: READREG at 0x0000 (magic register)
print("=== Test 1: READREG 0x0000 ===")
resp = u3v_readreg(dev, CTRL_OUT, CTRL_IN, 0x0000, 1, 1)
if resp:
    val = parse_readreg_response(resp)
    if val is not None:
        print(f"  >>> Register 0x0000 = 0x{val:08X}")

# Test 2: READREG at 0x0128 (device model)
print("\n=== Test 2: READREG 0x0128 ===")
resp = u3v_readreg(dev, CTRL_OUT, CTRL_IN, 0x0128, 1, 2)
if resp:
    val = parse_readreg_response(resp)

# Test 3: READREG at common GenICam registers
print("\n=== Test 3: READREG 0x0004 (heartbeat) ===")
resp = u3v_readreg(dev, CTRL_OUT, CTRL_IN, 0x0004, 1, 3)
if resp:
    val = parse_readreg_response(resp)

# Test 4: READREG at 0x0800 (device info)
print("\n=== Test 4: READREG 0x0800 ===")
resp = u3v_readreg(dev, CTRL_OUT, CTRL_IN, 0x0800, 1, 4)
if resp:
    val = parse_readreg_response(resp)

usb.util.release_interface(dev, 0)
print("\nDone!")