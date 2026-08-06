"""Direct USB3 Vision camera driver using PyUSB - bypasses broken libusb0 driver."""
import usb.core
import usb.util
import struct
import time
import logging
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Zeiss Axiocam 208 USB IDs
VENDOR_ID = 0x0758
PRODUCT_ID = 0x6002

# U3V (USB3 Vision) constants
U3V_CMD_GET_INFO = 0x0001
U3V_CMD_READ_MEM = 0x0002
U3V_CMD_WRITE_MEM = 0x0003
U3V_CMD_READ_MEM_ACK = 0x0004
U3V_CMD_WRITE_MEM_ACK = 0x0005
U3V_CMD_EVENT_NOTIFICATION = 0x0006
U3V_CMD_EVENT_ACK = 0x0007
U3V_CMD_ACTION = 0x0008
U3V_CMD_ACTION_ACK = 0x0009

# Endpoints from device descriptor
EP_BULK_IN_1 = 0x81
EP_BULK_OUT_1 = 0x01
EP_BULK_IN_2 = 0x82
EP_BULK_OUT_2 = 0x02


class U3VDirectCamera:
    """Direct USB3 Vision camera driver using PyUSB."""
    
    def __init__(self):
        self.dev: Optional[usb.core.Device] = None
        self.connected = False
        self.width = 0
        self.height = 0
        self.pixel_format = 0
        
    def connect(self) -> bool:
        """Connect to camera via PyUSB."""
        try:
            # Find camera
            self.dev = usb.core.find(idVendor=VENDOR_ID, idProduct=PRODUCT_ID)
            if self.dev is None:
                logger.error("Camera not found")
                return False
            
            logger.info(f"Found camera: {self.dev.product}")
            
            # Detach kernel driver if active (may fail if driver is in use)
            try:
                if self.dev.is_kernel_driver_active(0):
                    logger.info("Detaching kernel driver...")
                    self.dev.detach_kernel_driver(0)
            except Exception as e:
                logger.warning(f"Could not detach kernel driver (may be in use): {e}")
            
            # Set configuration
            self.dev.set_configuration()
            logger.info("Configuration set")
            
            # Get camera info
            self._get_camera_info()
            
            self.connected = True
            logger.info(f"Connected: {self.width}x{self.height}")
            return True
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def _get_camera_info(self):
        """Get camera information via U3V commands."""
        try:
            # Send GET_INFO command
            cmd = struct.pack('<H', U3V_CMD_GET_INFO)
            self.dev.write(EP_BULK_OUT_1, cmd, timeout=1000)
            
            # Read response
            response = self.dev.read(EP_BULK_IN_1, 512, timeout=1000)
            
            # Parse response (simplified - actual parsing depends on camera protocol)
            if len(response) >= 4:
                # Assume response contains width/height info
                # This is a placeholder - actual U3V protocol is more complex
                self.width = 2080  # Axiocam 208 typical resolution
                self.height = 1544
                
        except Exception as e:
            logger.warning(f"Could not get camera info: {e}")
            # Use default values
            self.width = 2080
            self.height = 1544
    
    def capture(self) -> Optional[np.ndarray]:
        """Capture a frame from the camera."""
        if not self.connected or self.dev is None:
            return None
        
        try:
            # Send acquisition start command
            cmd = struct.pack('<H', 0x0010)  # ACQUIRE_START
            self.dev.write(EP_BULK_OUT_1, cmd, timeout=1000)
            
            # Read image data
            # This is simplified - actual implementation needs proper U3V protocol handling
            image_size = self.width * self.height * 3  # RGB
            
            # Try to read from bulk endpoint
            data = self.dev.read(EP_BULK_IN_1, image_size, timeout=5000)
            
            if len(data) > 0:
                # Convert to numpy array
                img_array = np.frombuffer(data, dtype=np.uint8)
                img_array = img_array.reshape((self.height, self.width, 3))
                return img_array
            
            return None
            
        except Exception as e:
            logger.error(f"Capture failed: {e}")
            return None
    
    def disconnect(self):
        """Disconnect from camera."""
        if self.dev is not None:
            try:
                # Send acquisition stop command
                cmd = struct.pack('<H', 0x0011)  # ACQUIRE_STOP
                self.dev.write(EP_BULK_OUT_1, cmd, timeout=1000)
            except:
                pass
            
            # Release device
            usb.util.dispose_resources(self.dev)
            self.dev = None
        
        self.connected = False
        logger.info("Disconnected")
    
    def get_camera_info(self) -> dict:
        """Get camera information."""
        return {
            "name": "Zeiss Axiocam 208 (Direct USB)",
            "backend": "PyUSB Direct",
            "width": self.width,
            "height": self.height,
        }
    
    def set_exposure(self, exposure_us: float):
        """Set exposure time (placeholder)."""
        logger.info(f"Set exposure: {exposure_us} us (not implemented)")
    
    def set_gain(self, gain_db: float):
        """Set gain (placeholder)."""
        logger.info(f"Set gain: {gain_db} dB (not implemented)")


# Test function
def test_direct_camera():
    """Test direct USB camera access."""
    cam = U3VDirectCamera()
    
    print("Testing direct USB camera access...")
    if cam.connect():
        print(f"[OK] Connected: {cam.get_camera_info()}")
        
        print("Attempting to capture frame...")
        img = cam.capture()
        if img is not None:
            print(f"[OK] Captured image: {img.shape}")
        else:
            print("[FAIL] Capture failed")
        
        cam.disconnect()
    else:
        print("[FAIL] Connection failed")


if __name__ == "__main__":
    test_direct_camera()