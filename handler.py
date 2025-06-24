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
VERSION = "v25-thumbnail"

class ThumbnailProcessorV25:
    """v25 Thumbnail Processor - Google Script Compatible"""
    
    def __init__(self):
        print(f"[{VERSION}] Initializing - Google Script Compatible Version")
    
    def remove_black_box_numpy(self, image):
        """NumPy nonzero method - fastest and most effective"""
        img_np = np.array(image)
        h, w = img_np.shape[:2]
        print(f"[{VERSION}] NumPy method - Processing {w}x{h} image")
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Find non-black pixels (threshold > 30)
        threshold = 30
        non_black_pixels = gray > threshold
        
        # Find the bounding box of non-black pixels
        rows = np.any(non_black_pixels, axis=1)
        cols = np.any(non_black_pixels, axis=0)
        
        if np.any(rows) and np.any(cols):
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            
            # Add small margin
            margin = 10
            ymin = max(0, ymin - margin)
            ymax = min(h, ymax + margin + 1)
            xmin = max(0, xmin - margin)
            xmax = min(w, xmax + margin + 1)
            
            print(f"[{VERSION}] Black box detected: cropping to ({xmin},{ymin}) - ({xmax},{ymax})")
            
            # Crop the image
            cropped = img_np[ymin:ymax, xmin:xmax]
            return Image.fromarray(cropped), True
        
        return image, False
    
    def remove_black_box_pil(self, image):
        """PIL ImageChops method - backup"""
        try:
            print(f"[{VERSION}] PIL ImageChops method")
            
            # Get the background color (usually black at corners)
            bg_color = image.getpixel((0, 0))
            
            # Create a background image with that color
            bg = Image.new(image.mode, image.size, bg_color)
            
            # Calculate difference
            diff = ImageChops.difference(image, bg)
            
            # Add the difference to itself to amplify
            diff = ImageChops.add(diff, diff, 2.0, -100)
            
            # Get bounding box
            bbox = diff.getbbox()
            
            if bbox:
                # Add margin
                x1, y1, x2, y2 = bbox
                margin = 10
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(image.size[0], x2 + margin)
                y2 = min(image.size[1], y2 + margin)
                
                print(f"[{VERSION}] PIL bbox found: ({x1},{y1}) - ({x2},{y2})")
                return image.crop((x1, y1, x2, y2)), True
                
        except Exception as e:
            print(f"[{VERSION}] PIL method failed: {e}")
        
        return image, False
    
    def remove_black_box_threshold(self, image):
        """Threshold + Contour method - third option"""
        img_np = np.array(image)
        h, w = img_np.shape[:2]
        print(f"[{VERSION}] Threshold method")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get bounding rect of all contours
            x, y, w, h = cv2.boundingRect(np.concatenate(contours))
            
            # Add margin
            margin = 10
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(img_np.shape[1] - x, w + 2 * margin)
            h = min(img_np.shape[0] - y, h + 2 * margin)
            
            print(f"[{VERSION}] Contour bbox: ({x},{y}) size {w}x{h}")
            
            cropped = img_np[y:y+h, x:x+w]
            return Image.fromarray(cropped), True
        
        return image, False
    
    def apply_simple_enhancement(self, image):
        """Enhancement similar to v25 enhancement handler"""
        # 1. Brightness
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.15)  # Match v25 enhancement
        
        # 2. Contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.08)  # Match v25 enhancement
        
        # 3. Saturation
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.03)  # Match v25 enhancement
        
        # 4. Background color without shadow
        img_np = np.array(image)
        h, w = img_np.shape[:2]
        
        background_color = (250, 248, 245)
        brightness_overlay = np.full((h, w, 3), background_color, dtype=np.float32)
        
        # Uniform blending
        for i in range(3):
            img_np[:, :, i] = img_np[:, :, i] * 0.9 + brightness_overlay[:, :, i] * 0.1
        
        # Additional brightness
        img_np = np.clip(img_np * 1.02, 0, 255)
        
        return Image.fromarray(img_np.astype(np.uint8))
    
    def create_thumbnail_1000x1300(self, image):
        """Create exact 1000x1300 thumbnail - wedding ring centered and large"""
        target_size = (1000, 1300)
        
        # First find wedding ring area for tight crop
        img_np = np.array(image)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Find bright areas (rings)
        _, bright = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        combined = cv2.bitwise_or(bright, edges)
        
        # Noise removal
        kernel = np.ones((5, 5), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get bounding box of all contours
            all_points = np.concatenate(contours)
            x, y, w_box, h_box = cv2.boundingRect(all_points)
            
            # Only 5% padding for tighter crop
            padding_x = int(w_box * 0.05)
            padding_y = int(h_box * 0.05)
            
            x = max(0, x - padding_x)
            y = max(0, y - padding_y)
            w_box = min(img_np.shape[1] - x, w_box + 2 * padding_x)
            h_box = min(img_np.shape[0] - y, h_box + 2 * padding_y)
            
            # Crop
            cropped = image.crop((x, y, x + w_box, y + h_box))
        else:
            # Fallback: center 60% crop
            w, h = image.size
            margin_x = int(w * 0.2)
            margin_y = int(h * 0.2)
            cropped = image.crop((margin_x, margin_y, w - margin_x, h - margin_y))
        
        # Now resize to 1000x1300
        # Keep aspect ratio and fill
        cropped_w, cropped_h = cropped.size
        scale = max(1000 / cropped_w, 1300 / cropped_h)
        
        # Apply scale
        new_w = int(cropped_w * scale)
        new_h = int(cropped_h * scale)
        resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Create 1000x1300 canvas
        canvas = Image.new('RGB', target_size, (250, 248, 245))
        paste_x = (1000 - new_w) // 2
        paste_y = (1300 - new_h) // 2
        
        # Crop if needed
        if new_w > 1000 or new_h > 1300:
            # Center crop
            left = (new_w - 1000) // 2
            top = (new_h - 1300) // 2
            resized = resized.crop((left, top, left + 1000, top + 1300))
            canvas = resized
        else:
            canvas.paste(resized, (paste_x, paste_y))
        
        # Final sharpness increase
        enhancer = ImageEnhance.Sharpness(canvas)
        canvas = enhancer.enhance(1.8)  # Strong sharpening
        
        print(f"[{VERSION}] Created exact 1000x1300 thumbnail with tight crop")
        return canvas

def handler(job):
    """RunPod handler - V25 with Google Script compatibility"""
    print(f"[{VERSION}] ====== Thumbnail Handler Started ======")
    print(f"[{VERSION}] CRITICAL: Google Apps Script requires padding!")
    print(f"[{VERSION}] Make.com forbids padding - we remove it here")
    print(f"[{VERSION}] Google Script must add it back with:")
    print(f"[{VERSION}] while (base64Data.length % 4 !== 0) {{ base64Data += '='; }}")
    
    try:
        job_input = job.get("input", {})
        print(f"[{VERSION}] Input type: {type(job_input)}")
        print(f"[{VERSION}] Input keys: {list(job_input.keys()) if isinstance(job_input, dict) else 'Not a dict'}")
        
        # Find base64 image - FIXED with 'image_base64'
        base64_image = None
        
        if isinstance(job_input, dict):
            # CRITICAL: 'image_base64' as first key to check
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
        processor = ThumbnailProcessorV25()
        
        # Try multiple methods for black box removal
        had_black_box = False
        
        # Method 1: NumPy (fastest and most effective)
        image, removed = processor.remove_black_box_numpy(image)
        if removed:
            had_black_box = True
            print(f"[{VERSION}] Black box removed using NumPy method")
        else:
            # Method 2: PIL ImageChops
            image, removed = processor.remove_black_box_pil(image)
            if removed:
                had_black_box = True
                print(f"[{VERSION}] Black box removed using PIL method")
            else:
                # Method 3: Threshold + Contour
                image, removed = processor.remove_black_box_threshold(image)
                if removed:
                    had_black_box = True
                    print(f"[{VERSION}] Black box removed using threshold method")
        
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
        # But Google Apps Script NEEDS padding!
        # Google Script must add it back with:
        # while (base64Data.length % 4 !== 0) { base64Data += '='; }
        print(f"[{VERSION}] WARNING: Removing padding for Make.com")
        print(f"[{VERSION}] Google Apps Script MUST add padding back!")
        thumbnail_base64 = thumbnail_base64.rstrip('=')
        
        print(f"[{VERSION}] Thumbnail base64 length: {len(thumbnail_base64)}")
        print(f"[{VERSION}] Padding removed - Google Script must restore it!")
        
        # Return proper structure
        # RunPod wraps this in {"data": {"output": ...}}
        # Make.com path: {{4.data.output.output.thumbnail}}
        result = {
            "output": {
                "thumbnail": thumbnail_base64,
                "has_black_frame": had_black_box,
                "success": True,
                "version": VERSION,
                "thumbnail_size": [1000, 1300],
                "processing_method": "multi_method_v25",
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
    print("V25 - Google Script Compatible Version")
    print("CRITICAL: Padding is removed for Make.com")
    print("Google Apps Script MUST add padding back:")
    print("while (base64Data.length % 4 !== 0) { base64Data += '='; }")
    print("="*70)
    
    runpod.serverless.start({"handler": handler})
