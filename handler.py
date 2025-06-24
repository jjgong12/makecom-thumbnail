import runpod
import base64
import numpy as np
from PIL import Image, ImageEnhance, ImageChops
import cv2
import io
import os
import traceback
import time

# Version info
VERSION = "v26-thumbnail"

class ThumbnailProcessorV26:
    """v26 Thumbnail Processor - Aggressive Black Box Removal"""
    
    def __init__(self):
        print(f"[{VERSION}] Initializing - Aggressive Black Box Removal")
    
    def remove_black_box_aggressive(self, image):
        """Aggressive black box removal - multiple methods combined"""
        img_np = np.array(image)
        h, w = img_np.shape[:2]
        print(f"[{VERSION}] Aggressive removal - Processing {w}x{h} image")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Method 1: Find content area using multiple thresholds
        best_crop = None
        best_area = 0
        
        for threshold in [20, 30, 40, 50, 60, 70, 80]:
            # Find non-black pixels
            non_black = gray > threshold
            
            # Find bounding box
            rows = np.any(non_black, axis=1)
            cols = np.any(non_black, axis=0)
            
            if np.any(rows) and np.any(cols):
                ymin, ymax = np.where(rows)[0][[0, -1]]
                xmin, xmax = np.where(cols)[0][[0, -1]]
                
                # Calculate area
                area = (xmax - xmin) * (ymax - ymin)
                
                # Keep the smallest valid area (tightest crop)
                if area > 0 and (best_area == 0 or area < best_area * 0.95):
                    best_area = area
                    best_crop = (xmin, ymin, xmax, ymax)
                    print(f"[{VERSION}] Threshold {threshold}: Found crop {xmin},{ymin} to {xmax},{ymax}")
        
        # Method 2: Edge detection for black box
        if best_crop is None:
            # Use Canny edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get bounding rect of all contours
                all_points = np.concatenate(contours)
                x, y, w_box, h_box = cv2.boundingRect(all_points)
                best_crop = (x, y, x + w_box, y + h_box)
                print(f"[{VERSION}] Edge detection: Found crop {x},{y} to {x+w_box},{y+h_box}")
        
        # Method 3: Scan from edges inward
        if best_crop is None:
            # Scan from each edge
            threshold = 50
            
            # Top
            top = 0
            for i in range(h//2):
                if np.mean(gray[i, :]) > threshold:
                    top = i
                    break
            
            # Bottom
            bottom = h
            for i in range(h//2):
                if np.mean(gray[h-1-i, :]) > threshold:
                    bottom = h - i
                    break
            
            # Left
            left = 0
            for i in range(w//2):
                if np.mean(gray[:, i]) > threshold:
                    left = i
                    break
            
            # Right
            right = w
            for i in range(w//2):
                if np.mean(gray[:, w-1-i]) > threshold:
                    right = w - i
                    break
            
            if top > 0 or bottom < h or left > 0 or right < w:
                best_crop = (left, top, right, bottom)
                print(f"[{VERSION}] Edge scan: Found crop {left},{top} to {right},{bottom}")
        
        # Apply crop if found
        if best_crop:
            xmin, ymin, xmax, ymax = best_crop
            
            # Add small margin
            margin = 5  # Smaller margin for tighter crop
            xmin = max(0, xmin - margin)
            ymin = max(0, ymin - margin)
            xmax = min(w, xmax + margin)
            ymax = min(h, ymax + margin)
            
            print(f"[{VERSION}] Final crop: ({xmin},{ymin}) to ({xmax},{ymax})")
            
            # Crop the image
            cropped = img_np[ymin:ymax, xmin:xmax]
            return Image.fromarray(cropped), True
        
        print(f"[{VERSION}] No black box detected")
        return image, False
    
    def apply_simple_enhancement(self, image):
        """Enhancement matching v26 enhancement handler"""
        # 1. Brightness
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.25)  # Match v26
        
        # 2. Contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.12)  # Match v26
        
        # 3. Saturation
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(0.98)  # Match v26
        
        # 4. Background whitening
        img_np = np.array(image)
        h, w = img_np.shape[:2]
        
        background_color = (252, 251, 250)
        brightness_overlay = np.full((h, w, 3), background_color, dtype=np.float32)
        
        # Uniform blending
        for i in range(3):
            img_np[:, :, i] = img_np[:, :, i] * 0.85 + brightness_overlay[:, :, i] * 0.15
        
        # Additional brightness
        img_np = np.clip(img_np * 1.05, 0, 255)
        
        # Gamma correction
        gamma = 0.9
        img_np = np.power(img_np / 255.0, gamma) * 255
        img_np = np.clip(img_np, 0, 255)
        
        return Image.fromarray(img_np.astype(np.uint8))
    
    def create_thumbnail_1000x1300(self, image):
        """Create exact 1000x1300 thumbnail - extra tight crop on rings"""
        target_size = (1000, 1300)
        
        # Find wedding ring area
        img_np = np.array(image)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Multiple methods to find ring
        # Method 1: Bright areas (rings are usually bright)
        _, bright = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        
        # Method 2: Edge detection
        edges = cv2.Canny(gray, 30, 100)
        
        # Combine methods
        combined = cv2.bitwise_or(bright, edges)
        
        # Clean up noise
        kernel = np.ones((3, 3), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Filter small contours
            min_area = 100
            valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
            
            if valid_contours:
                # Get bounding box of all valid contours
                all_points = np.concatenate(valid_contours)
                x, y, w_box, h_box = cv2.boundingRect(all_points)
                
                # Very tight padding (2% only)
                padding_x = int(w_box * 0.02)
                padding_y = int(h_box * 0.02)
                
                x = max(0, x - padding_x)
                y = max(0, y - padding_y)
                w_box = min(img_np.shape[1] - x, w_box + 2 * padding_x)
                h_box = min(img_np.shape[0] - y, h_box + 2 * padding_y)
                
                # Crop
                cropped = image.crop((x, y, x + w_box, y + h_box))
            else:
                # Fallback: center 50% crop (tighter)
                w, h = image.size
                margin_x = int(w * 0.25)
                margin_y = int(h * 0.25)
                cropped = image.crop((margin_x, margin_y, w - margin_x, h - margin_y))
        else:
            # Fallback: aggressive center crop
            w, h = image.size
            margin_x = int(w * 0.3)
            margin_y = int(h * 0.3)
            cropped = image.crop((margin_x, margin_y, w - margin_x, h - margin_y))
        
        # Resize to fill 1000x1300
        cropped_w, cropped_h = cropped.size
        scale = max(1000 / cropped_w, 1300 / cropped_h)
        
        # Apply scale
        new_w = int(cropped_w * scale)
        new_h = int(cropped_h * scale)
        resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Create 1000x1300 canvas
        canvas = Image.new('RGB', target_size, (252, 251, 250))
        
        # Center crop if needed
        if new_w > 1000 or new_h > 1300:
            # Center crop
            left = (new_w - 1000) // 2
            top = (new_h - 1300) // 2
            resized = resized.crop((left, top, left + 1000, top + 1300))
            canvas = resized
        else:
            # Center paste
            paste_x = (1000 - new_w) // 2
            paste_y = (1300 - new_h) // 2
            canvas.paste(resized, (paste_x, paste_y))
        
        # Strong sharpness increase
        enhancer = ImageEnhance.Sharpness(canvas)
        canvas = enhancer.enhance(2.0)  # Very strong sharpening
        
        print(f"[{VERSION}] Created exact 1000x1300 thumbnail with extra tight crop")
        return canvas

def handler(job):
    """RunPod handler - V26 with aggressive black box removal"""
    print(f"[{VERSION}] ====== Thumbnail Handler Started ======")
    print(f"[{VERSION}] CRITICAL: Google Apps Script requires padding!")
    print(f"[{VERSION}] Make.com forbids padding - we remove it here")
    
    try:
        job_input = job.get("input", {})
        print(f"[{VERSION}] Input type: {type(job_input)}")
        print(f"[{VERSION}] Input keys: {list(job_input.keys()) if isinstance(job_input, dict) else 'Not a dict'}")
        
        # Find base64 image
        base64_image = None
        
        if isinstance(job_input, dict):
            # Check for image in various keys
            for key in ['image_base64', 'image', 'base64', 'data', 'input', 'file', 'imageData']:
                if key in job_input:
                    value = job_input[key]
                    if isinstance(value, str) and len(value) > 100:
                        base64_image = value
                        print(f"[{VERSION}] Found image in key: {key}")
                        break
        
        if not base64_image and isinstance(job_input, dict):
            for key, value in job_input.items():
                if isinstance(value, dict):
                    for sub_key in ['image_base64', 'image', 'base64', 'data']:
                        if sub_key in value and isinstance(value[sub_key], str) and len(value[sub_key]) > 100:
                            base64_image = value[sub_key]
                            print(f"[{VERSION}] Found image in nested: {key}.{sub_key}")
                            break
        
        if not base64_image and isinstance(job_input, str) and len(job_input) > 100:
            base64_image = job_input
        
        if not base64_image:
            return {
                "output": {
                    "thumbnail": None,
                    "error": "No image data found",
                    "success": False,
                    "version": VERSION,
                    "debug_info": {
                        "input_keys": list(job_input.keys()) if isinstance(job_input, dict) else [],
                        "first_key": list(job_input.keys())[0] if isinstance(job_input, dict) and job_input else None
                    }
                }
            }
        
        # Process image
        if ',' in base64_image and base64_image.startswith('data:'):
            base64_image = base64_image.split(',')[1]
        
        base64_image = base64_image.strip()
        
        # Add padding for decoding
        padding = 4 - len(base64_image) % 4
        if padding != 4:
            base64_image += '=' * padding
        
        # Decode
        img_data = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(img_data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        print(f"[{VERSION}] Image decoded: {image.size}")
        
        # Create processor
        processor = ThumbnailProcessorV26()
        
        # AGGRESSIVE black box removal
        image, had_black_box = processor.remove_black_box_aggressive(image)
        
        if had_black_box:
            print(f"[{VERSION}] Black box removed successfully")
        else:
            print(f"[{VERSION}] No black box found")
        
        # Apply color enhancement
        image = processor.apply_simple_enhancement(image)
        
        # Create exact 1000x1300 thumbnail
        thumbnail = processor.create_thumbnail_1000x1300(image)
        
        # Convert to base64
        buffer = io.BytesIO()
        thumbnail.save(buffer, format='PNG', quality=95)
        buffer.seek(0)
        
        thumbnail_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # CRITICAL FOR MAKE.COM: Remove padding
        print(f"[{VERSION}] WARNING: Removing padding for Make.com")
        print(f"[{VERSION}] Google Apps Script MUST add padding back!")
        thumbnail_base64 = thumbnail_base64.rstrip('=')
        
        print(f"[{VERSION}] Thumbnail base64 length: {len(thumbnail_base64)}")
        
        # Return proper structure
        result = {
            "output": {
                "thumbnail": thumbnail_base64,
                "has_black_frame": had_black_box,
                "success": True,
                "version": VERSION,
                "thumbnail_size": [1000, 1300],
                "processing_method": "aggressive_multi_method_v26",
                "warning": "Google Script must add padding: while (base64Data.length % 4 !== 0) { base64Data += '='; }"
            }
        }
        
        print(f"[{VERSION}] ====== Success - Returning Thumbnail ======")
        return result
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"[{VERSION}] ERROR: {error_msg}")
        traceback.print_exc()
        
        return {
            "output": {
                "thumbnail": None,
                "error": error_msg,
                "success": False,
                "version": VERSION
            }
        }

# RunPod serverless start
if __name__ == "__main__":
    print("="*70)
    print(f"Wedding Ring Thumbnail {VERSION}")
    print("V26 - Aggressive Black Box Removal")
    print("CRITICAL: Padding is removed for Make.com")
    print("Google Apps Script MUST add padding back:")
    print("while (base64Data.length % 4 !== 0) { base64Data += '='; }")
    print("="*70)
    
    runpod.serverless.start({"handler": handler})
