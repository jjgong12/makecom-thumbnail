import os
import io
import json
import base64
import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import time

VERSION = "v3"

def log_debug(message):
    """Debug logging with version info"""
    print(f"[{VERSION}] {message}")

def find_image_in_event(event):
    """Enhanced image finding with multiple strategies to prevent Exit Code 0"""
    try:
        log_debug("Starting enhanced image search")
        
        # Strategy 1: Direct input keys
        job_input = event.get("input", {})
        
        # Check common image keys
        possible_keys = ["image", "image_base64", "base64_image", "data", "img", "file"]
        
        for key in possible_keys:
            if key in job_input and job_input[key]:
                log_debug(f"Found image data in key: {key}")
                return job_input[key]
        
        # Strategy 2: Direct event keys (fallback)
        for key in possible_keys:
            if key in event and event[key]:
                log_debug(f"Found image data in event key: {key}")
                return event[key]
        
        # Strategy 3: String input (direct base64)
        if isinstance(job_input, str) and len(job_input) > 100:
            log_debug("Found string input as potential base64")
            return job_input
        
        # Strategy 4: Nested search
        for key, value in job_input.items():
            if isinstance(value, dict):
                for nested_key in possible_keys:
                    if nested_key in value and value[nested_key]:
                        log_debug(f"Found nested image data in: {key}.{nested_key}")
                        return value[nested_key]
        
        log_debug("No image data found in any strategy")
        return None
        
    except Exception as e:
        log_debug(f"Error in image search: {e}")
        return None

def decode_base64_image(base64_string):
    """Decode base64 image with enhanced error handling"""
    try:
        log_debug("Starting base64 decode process")
        
        # Remove whitespace and newlines
        base64_string = base64_string.strip().replace('\n', '').replace('\r', '').replace(' ', '')
        
        # Remove data URL prefix if present
        if 'base64,' in base64_string:
            base64_string = base64_string.split('base64,')[1]
            log_debug("Removed data URL prefix")
        
        # Clean any non-base64 characters
        import re
        base64_string = re.sub(r'[^A-Za-z0-9+/=]', '', base64_string)
        
        # Try standard decode first (without adding padding)
        try:
            image_data = base64.b64decode(base64_string, validate=True)
            log_debug("Standard decode successful")
        except Exception as e:
            log_debug(f"Standard decode failed: {e}, trying with padding")
            # Add padding if needed
            missing_padding = len(base64_string) % 4
            if missing_padding:
                base64_string += '=' * (4 - missing_padding)
            image_data = base64.b64decode(base64_string, validate=True)
            log_debug("Decode with padding successful")
        
        # Convert to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image data")
            
        log_debug(f"Image decoded successfully: {image.shape}")
        return image
        
    except Exception as e:
        log_debug(f"Base64 decode error: {e}")
        raise

def apply_enhancement_color_correction(image):
    """Apply same color correction as Enhancement Handler"""
    try:
        log_debug("Applying Enhancement Handler color correction")
        
        # Convert to RGB for PIL processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Step 1: Brightness boost (25% increase like Enhancement Handler)
        brightness_enhancer = ImageEnhance.Brightness(pil_image)
        enhanced = brightness_enhancer.enhance(1.25)
        log_debug("Applied 25% brightness boost")
        
        # Step 2: Contrast increase for clarity (10%)
        contrast_enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = contrast_enhancer.enhance(1.1)
        log_debug("Applied contrast enhancement")
        
        # Step 3: Reduce saturation slightly for clean look (5%)
        color_enhancer = ImageEnhance.Color(enhanced)
        enhanced = color_enhancer.enhance(0.95)
        log_debug("Applied saturation reduction")
        
        # Convert back to numpy for further processing
        enhanced_array = np.array(enhanced)
        enhanced_bgr = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2BGR)
        
        # Step 4: Apply white overlay (15% like Enhancement Handler)
        white_overlay = np.full_like(enhanced_bgr, 255, dtype=np.uint8)
        enhanced_bgr = cv2.addWeighted(enhanced_bgr, 0.85, white_overlay, 0.15, 0)
        log_debug("Applied 15% white overlay")
        
        return enhanced_bgr
        
    except Exception as e:
        log_debug(f"Color correction error: {e}")
        return image

def detect_and_crop_ring(image):
    """Enhanced ring detection and cropping for 1000x1300 thumbnail"""
    try:
        log_debug("Starting enhanced ring detection")
        
        h, w = image.shape[:2]
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Contour-based detection
        # Apply threshold to find ring edges
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        ring_bbox = None
        if contours:
            # Find largest contour (likely the ring)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding box
            x, y, cw, ch = cv2.boundingRect(largest_contour)
            
            # Validate contour size (should be significant portion of image)
            contour_area = cv2.contourArea(largest_contour)
            image_area = h * w
            
            if contour_area > image_area * 0.05:  # At least 5% of image
                ring_bbox = (x, y, cw, ch)
                log_debug(f"Ring detected via contours: {ring_bbox}")
        
        # Method 2: Fallback - use center region if contour detection fails
        if ring_bbox is None:
            log_debug("Contour detection failed, using center region")
            # Use center 70% of the image as ring area
            margin_x = int(w * 0.15)
            margin_y = int(h * 0.15)
            ring_bbox = (margin_x, margin_y, w - 2*margin_x, h - 2*margin_y)
        
        x, y, cw, ch = ring_bbox
        
        # Expand bbox for better framing (add 20% padding)
        expand_factor = 1.2
        center_x = x + cw // 2
        center_y = y + ch // 2
        new_w = int(cw * expand_factor)
        new_h = int(ch * expand_factor)
        
        # Calculate new coordinates
        new_x = max(0, center_x - new_w // 2)
        new_y = max(0, center_y - new_h // 2)
        new_x2 = min(w, new_x + new_w)
        new_y2 = min(h, new_y + new_h)
        
        # Crop the ring area
        ring_crop = image[new_y:new_y2, new_x:new_x2]
        
        log_debug(f"Ring cropped: original {ring_bbox}, final crop area: ({new_x}, {new_y}, {new_x2-new_x}, {new_y2-new_y})")
        
        return ring_crop, ring_bbox
        
    except Exception as e:
        log_debug(f"Ring detection error: {e}")
        # Fallback: use center 80% of image
        margin_x = int(w * 0.1)
        margin_y = int(h * 0.1)
        ring_crop = image[margin_y:h-margin_y, margin_x:w-margin_x]
        return ring_crop, (margin_x, margin_y, w-2*margin_x, h-2*margin_y)

def apply_light_detail_enhancement(image):
    """Apply light detail enhancement as requested"""
    try:
        log_debug("Applying light detail enhancement")
        
        # Light noise reduction
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 3, 3, 7, 21)
        
        # Light sharpening using unsharp mask
        gaussian = cv2.GaussianBlur(denoised, (0, 0), 1.0)
        sharpened = cv2.addWeighted(denoised, 1.3, gaussian, -0.3, 0)
        
        # Light CLAHE for local contrast
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        log_debug("Light detail enhancement completed")
        return enhanced
        
    except Exception as e:
        log_debug(f"Detail enhancement error: {e}")
        return image

def create_1000x1300_thumbnail(ring_crop):
    """Create optimized 1000x1300 thumbnail like Image 2"""
    try:
        log_debug("Creating 1000x1300 thumbnail")
        
        target_w, target_h = 1000, 1300
        crop_h, crop_w = ring_crop.shape[:2]
        
        # Calculate scaling to fill most of the thumbnail (90% usage)
        scale_x = (target_w * 0.9) / crop_w
        scale_y = (target_h * 0.9) / crop_h
        scale = min(scale_x, scale_y)
        
        # Resize with high quality
        new_w = int(crop_w * scale)
        new_h = int(crop_h * scale)
        
        resized = cv2.resize(ring_crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Create white background (like Image 2)
        thumbnail = np.ones((target_h, target_w, 3), dtype=np.uint8) * 248  # Slightly off-white
        
        # Center the ring in thumbnail
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        # Ensure we don't go out of bounds
        y_end = min(y_offset + new_h, target_h)
        x_end = min(x_offset + new_w, target_w)
        
        thumbnail[y_offset:y_end, x_offset:x_end] = resized[:y_end-y_offset, :x_end-x_offset]
        
        log_debug(f"Thumbnail created: ring size {new_w}x{new_h} in {target_w}x{target_h}")
        return thumbnail
        
    except Exception as e:
        log_debug(f"Thumbnail creation error: {e}")
        # Fallback: simple resize
        return cv2.resize(ring_crop, (1000, 1300), interpolation=cv2.INTER_LANCZOS4)

def image_to_base64(image):
    """Convert image to base64 string without padding (Make.com compatible)"""
    try:
        _, buffer = cv2.imencode('.png', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        # CRITICAL: Remove padding for Make.com compatibility
        img_base64 = img_base64.rstrip('=')
        log_debug("Image converted to base64 (padding removed)")
        return img_base64
    except Exception as e:
        log_debug(f"Base64 conversion error: {e}")
        raise

def handler(event):
    """Main handler function for thumbnail generation"""
    try:
        log_debug("=== Thumbnail Handler v3 Started ===")
        log_debug("Goal: Convert Image 1 → Image 2 style (ring enlargement + 1000x1300)")
        
        # Enhanced image finding to prevent Exit Code 0
        base64_image = find_image_in_event(event)
        
        if not base64_image:
            log_debug("No image provided - returning error response")
            return {
                "output": {
                    "error": "No image provided",
                    "status": "error",
                    "version": VERSION,
                    "timestamp": int(time.time())
                }
            }
        
        log_debug(f"Base64 string length: {len(base64_image)}")
        log_debug(f"Base64 string start: {base64_image[:100]}...")
        
        # Decode image
        image = decode_base64_image(base64_image)
        log_debug(f"Image decoded: {image.shape}")
        
        # Apply Enhancement Handler color correction (same as enhance endpoint)
        color_corrected = apply_enhancement_color_correction(image)
        log_debug("Enhancement Handler color correction applied")
        
        # Detect and crop ring for thumbnail
        ring_crop, ring_bbox = detect_and_crop_ring(color_corrected)
        log_debug("Ring detection and cropping completed")
        
        # Apply light detail enhancement
        enhanced_ring = apply_light_detail_enhancement(ring_crop)
        log_debug("Light detail enhancement applied")
        
        # Create 1000x1300 thumbnail (like Image 2)
        thumbnail = create_1000x1300_thumbnail(enhanced_ring)
        log_debug("1000x1300 thumbnail created")
        
        # Convert to base64
        thumbnail_base64 = image_to_base64(thumbnail)
        
        # Prepare response with nested output structure for Make.com
        response = {
            "output": {
                "thumbnail": thumbnail_base64,
                "processing_info": {
                    "version": VERSION,
                    "style": "Image_2_ring_enlargement",
                    "ring_bbox": ring_bbox,
                    "thumbnail_size": "1000x1300",
                    "color_correction": "Enhancement_Handler_style",
                    "detail_enhancement": "light",
                    "timestamp": int(time.time())
                }
            }
        }
        
        log_debug("=== Thumbnail Handler v3 Completed Successfully ===")
        return response
        
    except Exception as e:
        log_debug(f"Handler error: {e}")
        
        # Return error response with same structure (prevents Exit Code 0)
        return {
            "output": {
                "error": str(e),
                "status": "error",
                "version": VERSION,
                "timestamp": int(time.time())
            }
        }

# RunPod serverless handler
if __name__ == "__main__":
    log_debug("Starting RunPod serverless handler")
    runpod.serverless.start({"handler": handler})
