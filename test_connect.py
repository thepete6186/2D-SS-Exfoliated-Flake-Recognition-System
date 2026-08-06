#!/usr/bin/env python3
import subprocess
import time
import os
import logging

logging.basicConfig(level=logging.INFO)

print("[+] Stopping background processes using libusb0...")
subprocess.run(["taskkill", "/F", "/IM", "Labscope.exe"], capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
subprocess.run(["taskkill", "/F", "/IM", "LabscopeService.exe"], capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
subprocess.run(["taskkill", "/F", "/IM", "ZEN.exe"], capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)

# Pause to allow Windows to clear stale libusb0 handle tables
time.sleep(2)

from camera.zeiss_camera import ZeissCamera, _find_gentl_producers

print("[+] Checking GenTL CTI files...")
producers = _find_gentl_producers()
print(f"[+] Loaded CTI Producer: {producers}")

print("[+] Connecting to Axiocam 208 via GenTL...")
cam = ZeissCamera(backend="gentl")

if cam.connect():
    print("\n" + "="*50)
    print("[SUCCESS] Connected to Zeiss Axiocam 208!")
    info = cam.get_camera_info()
    print(f"[INFO] Camera Model: {info.get('model', 'Axiocam 208')}")
    print(f"[INFO] Resolution: {info.get('resolution')}")
    
    frame = cam.capture()
    if frame is not None:
        print(f"[SUCCESS] Captured Frame Shape: {frame.shape}")
    
    cam.disconnect()
    print("[+] Connection closed cleanly.")
    print("="*50 + "\n")
else:
    print("\n[-] GenTL enumerated 0 devices.")
    print("[-] Action Required:")
    print("    Unplug and replug the Axiocam 208 USB 3.0 cable to reset the physical driver handle.")