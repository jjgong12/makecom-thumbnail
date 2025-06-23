import os
import sys
import runpod
import base64
import cv2
import numpy as np
import io
import traceback
import time

print("[Thumbnail v2] Starting Wedding Ring Thumbnail Handler")
print(f"[Thumbnail v2] Python version: {sys.version}")
print(f"[Thumbnail v2] OpenCV version: {cv2.__version__}")
print("="*70)

def remove_padding_safe(base64_string):
    """Remove padding from base64 string for Make.com compatibility"""
    return base64_string.rstrip('=')

def decode_base64_image(base64_string):
    """Decode base64 image with enhanced error handling"""
    try:
        # Clean the base64 string
        base64_string = base64_string.strip()
        
        # Remove data URL prefix if present
        if 'base64,' in base64_string:
            base64_string = base64_string.split('base64,')[1]
        
        # Try direct decode first
        try:
            image_data = base64.b64decode(base64_string)
        except:
            # Add padding if needed
            missing_padding = len(base64_string) % 4
            if missing_padding:
                base64_string += '=' * (4 - missing_padding)
            image_data = base64.b64decode(base64_string)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        return image
    except Exception as e:
        print(f"[Thumbnail v2] Error decoding image: {str(e)}")
        raise

def find_image_in_event(event):
    """Enhanced image finding with multiple fallback strategies"""
    print("[Thumbnail v2] Starting image search...")
    print(f"[Thumbnail v2] Event type: {type(event)}")
    print(f"[Thumbnail v2] Event keys: {list(event.keys()) if isinstance(event, dict) else 'Not a dict'}")
    
    # Strategy 1: Direct input
    input_data = event.get("input", {})
    print(f"[Thumbnail v2] Input type: {type(input_data)}")
    
    if isinstance(input_data, dict):
        print(f"[Thumbnail v2] Input keys: {list(input_data.keys())}")
        # Common keys
        for key in ['image', 'image_base64', 'base64', 'img', 'data']:
            if key in input_data and input_data[key]:
                print(f"[Thumbnail v2] Found image in input.{key}")
                return input_data[key]
    
    # Strategy 2: Direct event keys
    for key in ['image', 'image_base64', 'base64']:
        if key in event and event[key]:
            print(f"[Thumbnail v2] Found image in event.{key}")
            return event[key]
    
    # Strategy 3: String input
    if isinstance(input_data, str) and len(input_data) > 100:
        print("[Thumbnail v2] Input is string, assuming base64")
        return input_data
    
    # Strategy 4: Nested search
    if isinstance(input_data, dict):
        for key, value in input_data.items():
            if isinstance(value, str) and len(value) > 100:
                print(f"[Thumbnail v2] Found potential image in input.{key}")
                return value
    
    print("[Thumbnail v2] No image found in event")
    return None

def detect_metal_type(image):
    """Detect metal type based on color analysis"""
    # Convert to LAB color space for better color analysis
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Get center region
    h, w = image.shape[:2]
    center_y, center_x = h // 2, w // 2
    roi_size = min(h, w) // 4
    roi = lab[center_y-roi_size:center_y+roi_size, 
              center_x-roi_size:center_x+roi_size]
    
    # Calculate average LAB values
    avg_lab = np.mean(roi.reshape(-1, 3), axis=0)
    l, a, b = avg_lab
    
    # Detect metal type based on LAB values
    if b > 15:  # Yellow tones
        return "yellow_gold"
    elif a > 5:  # Red/pink tones
        return "rose_gold"
    elif l > 180:  # Very bright
        return "plain_white"
    else:
        return "white_gold"

def detect_and_crop_ring(image):
    """Detect ring and crop it optimally for thumbnail"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply threshold
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological operations to connect ring parts
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find largest contour
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        
        # Add generous padding for aesthetic
        pad = max(100, int(min(w, h) * 0.35))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image.shape[1], x + w + pad)
        y2 = min(image.shape[0], y + h + pad)
        
        return image[y1:y2, x1:x2], (x, y, w, h)
    
    return image, None

def create_high_quality_thumbnail(image, bbox=None):
    """Create a high-quality 1000x1300 thumbnail with upscaling and enhancement"""
    target_w, target_h = 1000, 1300
    
    # If we have a bounding box, use the cropped region
    if bbox is not None:
        ring_crop, _ = detect_and_crop_ring(image)
    else:
        ring_crop = image
    
    # Apply initial enhancement to the crop
    # Sharpen
    kernel = np.array([[-1,-1,-1],
                      [-1, 9,-1],
                      [-1,-1,-1]])
    sharpened = cv2.filter2D(ring_crop, -1, kernel)
    ring_crop = cv2.addWeighted(ring_crop, 0.7, sharpened, 0.3, 0)
    
    # Calculate scale to fit the ring optimally
    crop_h, crop_w = ring_crop.shape[:2]
    
    # We want the ring to occupy about 80% of the frame
    scale = min(target_w/crop_w, target_h/crop_h) * 0.8
    new_w = int(crop_w * scale)
    new_h = int(crop_h * scale)
    
    # Use INTER_CUBIC for upscaling (better quality than LANCZOS for upscaling)
    if scale > 1:
        # For upscaling, use INTER_CUBIC with multiple passes for better quality
        temp_scale = np.sqrt(scale)
        temp_w = int(crop_w * temp_scale)
        temp_h = int(crop_h * temp_scale)
        
        # First pass
        temp_resized = cv2.resize(ring_crop, (temp_w, temp_h), interpolation=cv2.INTER_CUBIC)
        
        # Second pass
        resized = cv2.resize(temp_resized, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    else:
        resized = cv2.resize(ring_crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Apply unsharp mask for detail enhancement
    gaussian = cv2.GaussianBlur(resized, (0, 0), 2.0)
    sharpened = cv2.addWeighted(resized, 1.5, gaussian, -0.5, 0)
    
    # Enhance contrast and brigh
