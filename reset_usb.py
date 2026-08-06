"""Disable and re-enable the Zeiss camera USB device to force clean re-enumeration."""
import subprocess
import time
import os

# Find the device instance ID
print("=== Finding camera device instance ===")
result = subprocess.run(
    ["pnputil", "/enum-devices"],
    capture_output=True, text=True, timeout=15,
    creationflags=subprocess.CREATE_NO_WINDOW
)
lines = result.stdout.split('\n')
instance_id = None
for i, line in enumerate(lines):
    if 'VID_0758&PID_6002' in line and 'Instance ID' in line:
        instance_id = line.split(':', 1)[1].strip()
        print(f"  Found instance: {instance_id}")
        break

if not instance_id:
    print("  Could not find camera instance ID!")
    exit(1)

# Disable the device
print(f"\n=== Disabling device: {instance_id} ===")
result = subprocess.run(
    ["pnputil", "/disable-device", instance_id],
    capture_output=True, text=True, timeout=15,
    creationflags=subprocess.CREATE_NO_WINDOW
)
print(f"  Disable result: {result.stdout.strip()}")
if result.returncode != 0:
    print(f"  Disable error: {result.stderr.strip()}")
    # Try with /remove-device
    print("  Trying /remove-device...")
    result = subprocess.run(
        ["pnputil", "/remove-device", instance_id],
        capture_output=True, text=True, timeout=15,
        creationflags=subprocess.CREATE_NO_WINDOW
    )
    print(f"  Remove result: {result.stdout.strip()}")

time.sleep(3)

# Re-enable the device
print(f"\n=== Re-enabling device: {instance_id} ===")
result = subprocess.run(
    ["pnputil", "/enable-device", instance_id],
    capture_output=True, text=True, timeout=15,
    creationflags=subprocess.CREATE_NO_WINDOW
)
print(f"  Enable result: {result.stdout.strip()}")
if result.returncode != 0:
    print(f"  Enable error: {result.stderr.strip()}")

time.sleep(3)

# Verify device is back
print("\n=== Verifying device ===")
result = subprocess.run(
    ["pnputil", "/enum-devices", "/instanceid", instance_id],
    capture_output=True, text=True, timeout=10,
    creationflags=subprocess.CREATE_NO_WINDOW
)
print(result.stdout)