# Camera Annotator - Setup Guide

## Current Status
- ✅ All software fixes applied
- ✅ Environment variables configured
- ✅ GenTL backend working (when camera accessible)
- ✅ Folder watch mode added for Labscope integration
- ⚠️ Labscope service auto-restarts and grabs camera

## The Problem
The Zeiss Axiocam 208 camera with libusb0 driver can only be accessed by **ONE application at a time**. When Labscope opens the camera, no other app can use it.

## Solutions

### Option 1: Folder Watch Mode (RECOMMENDED - Works with both apps)
1. Open Labscope
2. Configure Labscope to save images to a folder (e.g., `C:\Users\206D\Documents\Labscope\Images`)
3. Run camera annotator: `python camera/camera_annotator.py` or `launch_annotator.bat`
4. Click "Watch Labscope Folder" button
5. Select the same folder Labscope saves to
6. The annotator will automatically load new images as Labscope saves them

**Pros:**
- Both apps run simultaneously
- Near real-time workflow
- No fighting over camera

**Cons:**
- Not truly "live" (1-2 second delay)
- Depends on Labscope saving images

### Option 2: Use Apps Separately (One at a Time)
**To use Labscope:**
- Just open Labscope normally
- Don't run the Python script

**To use Camera Annotator:**
1. Close Labscope completely
2. Run: `launch_annotator.bat` (kills Labscope service automatically)
3. Click "Connect Camera"

**Pros:**
- True live feed in annotator
- Simple workflow

**Cons:**
- Cannot use both at same time
- Must close one to open the other

### Option 3: Disable Labscope Service Permanently (Requires Admin)
Have an administrator run this command:
```
sc config LabscopeFileTransferService start= disabled
sc stop LabscopeFileTransferService
```

After this:
- Labscope will NOT auto-start
- You can use annotator whenever you want
- Labscope only works when manually started
- You still cannot use both simultaneously

## If Camera Stops Working After Killing Labscope

**The driver may enter a bad state. Fix it by:**

1. **Unplug camera USB cable**
2. **Wait 10+ seconds** (critical - let camera fully power down)
3. **Plug camera back in**
4. **Wait 5 seconds** for Windows to detect it
5. Run `launch_annotator.bat` or open Labscope

If that doesn't work, **reboot your computer**.

## Why Simultaneous Live Feed Is Impossible

The Zeiss Axiocam 208 uses a **proprietary USB driver (libusb0)** that only allows **exclusive access**. This is enforced at the Windows kernel level and cannot be bypassed with software.

To enable simultaneous access, you would need:
1. **Aravis SDK** installed (requires admin + download)
2. **Usb3CamHS adapter** configured with Aravis
3. **Micro-Manager** running as a camera server
4. Both apps connecting to Micro-Manager server

This setup is complex and requires admin rights to install the Aravis SDK.

## Quick Start

**For Labscope only:**
- Just open Labscope

**For annotator only:**
- Double-click `launch_annotator.bat`

**For both (folder watch mode):**
1. Open Labscope, set save folder
2. Double-click `launch_annotator.bat`
3. Click "Watch Labscope Folder"
4. Select Labscope's save folder

## Troubleshooting

**"No devices enumerated" error:**
- Unplug camera, wait 10 seconds, plug back in
- Reboot if that doesn't work

**Labscope grabs camera when I run annotator:**
- Use folder watch mode instead
- Or get admin to disable Labscope service

**Annotator won't connect:**
- Make sure Labscope is completely closed
- Run `launch_annotator.bat` to kill Labscope service first
- Unplug/replug camera if needed

## Contact
For hardware/driver issues, contact Zeiss support.