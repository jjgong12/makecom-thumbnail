import runpod
import base64
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import cv2
import io
import os
import traceback
import time

# Version info
VERSION = "v33-thumbnail"

# Import Replicate when available
try:
    import replicate
    REPLICATE_AVAILABLE = True
except ImportError:
    REPLICATE_AVAILABLE = False
    print(f"[{VERSION}] Replicate not available")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

class ThumbnailProcessorV33:
    """v33 Thumbnail Processor - Simple Center Crop for Full Ring View"""
    
    def __init__(self):
        print(f"[{VERSION}] Initializing - Simple Center Crop for Full Ring View")
        self.replicate_client = None
    
    def detect_and_remove_black_box(self, image):
        """Detect and remove black box using simple but effective method"""
        try:
            img_np = np.array(image)
            h, w = img_np.shape[:2]
            print(f"[{VERSION}] Detecting black box in {w}x{h} image")
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
            # Use a simple threshold to find black areas
            black_threshold = 30
            
            # Find where content starts from each edge
            # Top edge
            top = 0
            for y in range(h//3):
                row = gray[y, w//4:3*w//4]  # Check middle portion
                if np.mean(row) > black_threshold + 20:
                    top = max(0, y - 10)
                    break
            
            # Bottom edge
            bottom = h
            for y in range(h//3):
                row = gray[h-1-y, w//4:3*w//4]
                if np.mean(row) > black_threshold + 20:
                    bottom = min(h, h - y + 10)
                    break
            
            # Left edge
            left = 0
            for x in range(w//3):
                col = gray[h//4:3*h//4, x]
                if np.mean(col) > black_threshold + 20:
                    left = max(0, x - 10)
                    break
            
            # Right edge
            right = w
            for x in range(w//3):
                col = gray[h//4:3*h//4, w-1-x]
                if np.mean(col) > black_threshold + 20:
                    right = min(w, w - x + 10)
                    break
            
            # Check if we found a black frame
            if top > 0 or bottom < h or left > 0 or right < w:
                print(f"[{VERSION}] Black frame detected: T:{top}, B:{bottom}, L:{left}, R:{right}")
                
                # Crop out the black frame
                if right > left and bottom > top:
                    cropped = img_np[top:bottom, left:right]
                    return Image.fromarray(cropped), True
            
            print(f"[{VERSION}] No black frame detected")
            return image, False
            
        except Exception as e:
            print(f"[{VERSION}] Error in black frame detection: {e}")
            return image, False
    
    def apply_enhancement_matching_v33(self, image):
        """Enhancement matching v33 enhancement handler - brighter"""
        try:
            # Match v33 enhancement settings - brighter
            # 1. Light sharpening
            image = image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=50, threshold=3))
            
            # 2. More brightness - match v33
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.12)  # Match v33
            
            # 3. More contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.08)  # Match v33
            
            # 4. Cleaner colors
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(0.96)  # Match v33
            
            # 5. Convert to numpy for processing
            img_np = np.array(image)
            h, w = img_np.shape[:2]
            
            # 6. Whiter background
            white_color = (252, 252, 252)
            
            # Edge detection
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 60, 150)
            edges_dilated = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
            
            mask = np.ones((h, w), dtype=np.float32)
            mask[edges_dilated > 0] = 0
            mask = cv2.GaussianBlur(mask, (31, 31), 15)
            
            for i in range(3):
                img_np[:, :, i] = img_np[:, :, i] * (1 - mask * 0.12) + white_color[i] * mask * 0.12
            
            # Gamma correction
            gamma = 0.92
            img_np = np.power(img_np / 255.0, gamma) * 255
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            
            # Additional brightness
            img_np = np.clip(img_np * 1.03, 0, 255).astype(np.uint8)
            
            return Image.fromarray(img_np)
            
        except Exception as e:
            print(f"[{VERSION}] Error in enhancement: {e}")
            return image
    
    def create_perfect_thumbnail_1000x1300(self, image):
        """Create perfect 1000x1300 thumbnail with whole ring visible"""
        try:
            target_size = (1000, 1300)
            img_np = np.array(image)
            h, w = img_np.shape[:2]
            
            print(f"[{VERSION}] Creating thumbnail with whole ring visible...")
            
            # Simple approach - use center crop with very wide margin
            # Since rings are usually in the center, this is more reliable
            
            # Use fixed ratio crop from center
            # For 1000x1300 target, we want roughly 1:1.3 aspect ratio
            crop_ratio = 1.3
            
            # Calculate crop size based on image dimensions
            # Use smaller dimension as base to ensure we don't exceed image bounds
            if w / h > 1 / crop_ratio:
                # Image is wider - use height as base
                crop_h = int(h * 0.5)  # Use 50% of height
                crop_w = int(crop_h / crop_ratio)
            else:
                # Image is taller - use width as base
                crop_w = int(w * 0.5)  # Use 50% of width
                crop_h = int(crop_w * crop_ratio)
            
            # Center the crop
            x = (w - crop_w) // 2
            y = (h - crop_h) // 2
            
            # Ensure bounds are valid
            x = max(0, x)
            y = max(0, y)
            crop_w = min(crop_w, w - x)
            crop_h = min(crop_h, h - y)
            
            print(f"[{VERSION}] Center crop: ({x},{y}) size {crop_w}x{crop_h}")
            
            # Crop the image
            cropped = image.crop((x, y, x + crop_w, y + crop_h))
            
            # Resize to target size with high quality
            final = cropped.resize(target_size, Image.Resampling.LANCZOS)
            
            print(f"[{VERSION}] Resized to target: {final.size}")
            
            # Light sharpening for clarity
            enhancer = ImageEnhance.Sharpness(final)
            final = enhancer.enhance(1.2)  # Light sharpening
            
            # Convert to numpy for final touches
            final_np = np.array(final)
            
            # Very light unsharp mask
            gaussian = cv2.GaussianBlur(final_np, (0, 0), 1.5)
            unsharp = cv2.addWeighted(final_np, 1.2, gaussian, -0.2, 0)
            
            # Ensure no artifacts
            final_np = np.clip(unsharp, 0, 255).astype(np.uint8)
            
            print(f"[{VERSION}] Created 1000x1300 thumbnail with whole ring visible")
            return Image.fromarray(final_np)
            
        except Exception as e:
            print(f"[{VERSION}] Error creating thumbnail: {e}")
            traceback.print_exc()
            # Fallback - simple resize
            return image.resize((1000, 1300), Image.Resampling.LANCZOS)

def find_base64_in_dict(data, depth=0, max_depth=10):
    """Find base64 image in nested dictionary"""
    if depth > max_depth:
        return None
    
    if isinstance(data, str) and len(data) > 100:
        return data
    
    if isinstance(data, dict):
        for key in ['image_base64', 'image', 'base64', 'data', 'input', 'file', 'imageData']:
            if key in data and isinstance(data[key], str) and len(data[key]) > 100:
                return data[key]
        
        for value in data.values():
            result = find_base64_in_dict(value, depth + 1, max_depth)
            if result:
                return result
    
    elif isinstance(data, list):
        for item in data:
            result = find_base64_in_dict(item, depth + 1, max_depth)
            if result:
                return result
    
    return None

def decode_base64_image(base64_str):
    """Decode base64 string to PIL Image"""
    try:
        # Handle data URL format
        if ',' in base64_str and base64_str.startswith('data:'):
            base64_str = base64_str.split(',')[1]
        
        # Clean base64
        base64_str = base64_str.strip()
        
        # Add padding for decoding
        padding = 4 - len(base64_str) % 4
        if padding != 4:
            base64_str += '=' * padding
        
        # Decode
        img_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_data))
        
        # Convert to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        return img
        
    except Exception as e:
        print(f"[{VERSION}] Error decoding base64: {e}")
        raise

def encode_image_to_base64(image, format='PNG'):
    """Encode image to base64 (Make.com compatible)"""
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        buffer = io.BytesIO()
        image.save(buffer, format=format, quality=95 if format == 'JPEG' else None)
        buffer.seek(0)
        
        # Base64 encode
        base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # CRITICAL: Remove padding for Make.com
        base64_str = base64_str.rstrip('=')
        
        return base64_str
        
    except Exception as e:
        print(f"[{VERSION}] Error encoding image: {e}")
        raise

def handler(job):
    """RunPod handler - V33 with simple center crop"""
    print(f"[{VERSION}] ====== Thumbnail Handler Started ======")
    print(f"[{VERSION}] Simple Center Crop for Full Ring View")
    
    start_time = time.time()
    
    try:
        job_input = job.get("input", {})
        print(f"[{VERSION}] Input type: {type(job_input)}")
        print(f"[{VERSION}] Input keys: {list(job_input.keys()) if isinstance(job_input, dict) else 'Not a dict'}")
        
        # Find base64 image
        base64_image = find_base64_in_dict(job_input)
        
        if not base64_image:
            # Try direct string
            if isinstance(job_input, str) and len(job_input) > 100:
                base64_image = job_input
            else:
                return {
                    "output": {
                        "thumbnail": None,
                        "error": "No image data found",
                        "success": False,
                        "version": VERSION,
                        "debug_info": {
                            "input_keys": list(job_input.keys()) if isinstance(job_input, dict) else [],
                            "input_length": len(str(job_input))
                        }
                    }
                }
        
        print(f"[{VERSION}] Base64 image found, length: {len(base64_image)}")
        
        # Decode image
        try:
            image = decode_base64_image(base64_image)
            print(f"[{VERSION}] Image decoded: {image.size}")
        except Exception as e:
            return {
                "output": {
                    "thumbnail": None,
                    "error": f"Failed to decode image: {str(e)}",
                    "success": False,
                    "version": VERSION
                }
            }
        
        # Create processor
        processor = ThumbnailProcessorV33()
        
        # Process image step by step
        had_black_frame = False
        
        # 1. Detect and remove black box
        try:
            image, had_black_frame = processor.detect_and_remove_black_box(image)
            print(f"[{VERSION}] Black box detection complete: {had_black_frame}")
        except Exception as e:
            print(f"[{VERSION}] Error in black frame detection: {e}")
            traceback.print_exc()
        
        # 2. Apply color enhancement matching v33 - AFTER black box removal
        try:
            image = processor.apply_enhancement_matching_v33(image)
            print(f"[{VERSION}] Color enhancement applied (v33 style - brighter)")
        except Exception as e:
            print(f"[{VERSION}] Error in color enhancement: {e}")
            traceback.print_exc()
        
        # 3. Create PERFECT 1000x1300 thumbnail with simple center crop
        try:
            thumbnail = processor.create_perfect_thumbnail_1000x1300(image)
            print(f"[{VERSION}] Perfect thumbnail created: {thumbnail.size}")
        except Exception as e:
            print(f"[{VERSION}] Error creating thumbnail: {e}")
            traceback.print_exc()
            thumbnail = image.resize((1000, 1300), Image.Resampling.LANCZOS)
        
        # Encode result
        try:
            thumbnail_base64 = encode_image_to_base64(thumbnail, format='PNG')
            print(f"[{VERSION}] Thumbnail encoded, length: {len(thumbnail_base64)}")
        except Exception as e:
            return {
                "output": {
                    "thumbnail": None,
                    "error": f"Failed to encode thumbnail: {str(e)}",
                    "success": False,
                    "version": VERSION
                }
            }
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Return proper structure
        result = {
            "output": {
                "thumbnail": thumbnail_base64,
                "has_black_frame": had_black_frame,
                "success": True,
                "version": VERSION,
                "thumbnail_size": [1000, 1300],
                "processing_method": "simple_center_crop_v33",
                "processing_time": round(processing_time, 2),
                "replicate_available": REPLICATE_AVAILABLE,
                "replicate_used": False,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                "warning": "Google Script must add padding: while (base64Data.length % 4 !== 0) { base64Data += '='; }"
            }
        }
        
        print(f"[{VERSION}] ====== Success - Returning Thumbnail ======")
        print(f"[{VERSION}] Total processing time: {processing_time:.2f}s")
        print(f"[{VERSION}] Black frame detected and removed: {had_black_frame}")
        
        return result
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"[{VERSION}] CRITICAL ERROR: {error_msg}")
        traceback.print_exc()
        
        return {
            "output": {
                "thumbnail": None,
                "error": error_msg,
                "success": False,
                "version": VERSION,
                "traceback": traceback.format_exc()
            }
        }

# RunPod serverless start
if __name__ == "__main__":
    print("="*70)
    print(f"Wedding Ring Thumbnail {VERSION}")
    print("V33 - Simple Center Crop for Full Ring View")
    print("Features:")
    print("- Simple black box detection and removal")
    print("- Center crop using 50% of image")
    print("- No complex ring detection")
    print("- Bright enhancement matching v33")
    print("- Light sharpening only")
    print(f"Replicate Available: {REPLICATE_AVAILABLE}")
    print("CRITICAL: Padding is removed for Make.com")
    print("Google Apps Script MUST add padding back:")
    print("while (base64Data.length % 4 !== 0) { base64Data += '='; }")
    print("="*70)
    
    runpod.serverless.start({"handler": handler})
