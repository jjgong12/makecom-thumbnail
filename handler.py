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
VERSION = "v34-thumbnail"

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

class ThumbnailProcessorV34:
    """v34 Thumbnail Processor - Enhanced Detail Processing with Noise/Dust/Scratch Removal"""
    
    def __init__(self):
        print(f"[{VERSION}] Initializing - Strong Detail Enhancement & Cleanup")
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
    
    def remove_noise_and_defects(self, image):
        """Advanced noise, dust, and scratch removal"""
        try:
            img_np = np.array(image)
            
            # 1. Initial denoising
            print(f"[{VERSION}] Applying advanced denoising...")
            denoised = cv2.fastNlMeansDenoisingColored(img_np, None, 3, 3, 7, 21)
            
            # 2. Dust and scratch removal using median filter
            print(f"[{VERSION}] Removing dust and scratches...")
            # Small kernel for tiny dust
            dust_removed = cv2.medianBlur(denoised, 3)
            
            # 3. Detail-preserving smoothing
            # Using bilateral filter to preserve edges while smoothing surfaces
            smooth = cv2.bilateralFilter(dust_removed, 5, 30, 30)
            
            # 4. Scratch detection and removal
            # Convert to grayscale for scratch detection
            gray = cv2.cvtColor(smooth, cv2.COLOR_RGB2GRAY)
            
            # Detect thin lines (potential scratches)
            edges = cv2.Canny(gray, 30, 60)
            
            # Morphological operations to identify scratch-like structures
            kernel_line = np.ones((3,1), np.uint8)
            scratches_v = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_line, iterations=1)
            kernel_line = np.ones((1,3), np.uint8)
            scratches_h = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_line, iterations=1)
            scratches = cv2.bitwise_or(scratches_v, scratches_h)
            
            # Dilate scratches slightly for inpainting
            scratches = cv2.dilate(scratches, np.ones((3,3), np.uint8), iterations=1)
            
            # Inpaint scratches
            if np.any(scratches):
                print(f"[{VERSION}] Inpainting detected scratches...")
                result = cv2.inpaint(smooth, scratches, 3, cv2.INPAINT_TELEA)
            else:
                result = smooth
            
            # 5. Final touch - very light gaussian to ensure smoothness
            result = cv2.GaussianBlur(result, (3, 3), 0.5)
            
            # Blend with original to preserve some texture
            result = cv2.addWeighted(img_np, 0.3, result, 0.7, 0)
            
            return Image.fromarray(result)
            
        except Exception as e:
            print(f"[{VERSION}] Error in noise/defect removal: {e}")
            return image
    
    def apply_enhancement_matching_v34(self, image):
        """Enhancement matching v34 - stronger detail enhancement"""
        try:
            # First apply noise and defect removal
            image = self.remove_noise_and_defects(image)
            
            # 1. Strong sharpening for detail
            print(f"[{VERSION}] Applying strong detail enhancement...")
            image = image.filter(ImageFilter.UnsharpMask(radius=1.5, percent=100, threshold=2))
            
            # 2. Brightness boost
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.15)  # Brighter
            
            # 3. Strong contrast for detail
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.12)  # More contrast
            
            # 4. Clean colors
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(0.98)
            
            # 5. Convert to numpy for advanced processing
            img_np = np.array(image)
            h, w = img_np.shape[:2]
            
            # 6. Advanced detail enhancement using high-pass filter
            # Create high-pass filter
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (21, 21), 0)
            high_pass = cv2.subtract(gray, blur)
            high_pass = cv2.normalize(high_pass, None, 0, 50, cv2.NORM_MINMAX)
            
            # Add high-pass to each channel for detail enhancement
            for i in range(3):
                img_np[:, :, i] = cv2.add(img_np[:, :, i], high_pass)
            
            # 7. Clean white background
            white_color = (252, 252, 252)
            
            # Edge detection for background
            edges = cv2.Canny(gray, 60, 150)
            edges_dilated = cv2.dilate(edges, np.ones((5,5), np.uint8), iterations=2)
            
            mask = np.ones((h, w), dtype=np.float32)
            mask[edges_dilated > 0] = 0
            mask = cv2.GaussianBlur(mask, (51, 51), 25)
            
            # Apply white background
            for i in range(3):
                img_np[:, :, i] = img_np[:, :, i] * (1 - mask * 0.15) + white_color[i] * mask * 0.15
            
            # 8. Final clarity boost
            # Gamma correction for better clarity
            gamma = 0.9
            img_np = np.power(img_np / 255.0, gamma) * 255
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            
            # 9. Micro-contrast enhancement
            # Using CLAHE on L channel
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            img_np = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
            
            return Image.fromarray(img_np)
            
        except Exception as e:
            print(f"[{VERSION}] Error in enhancement: {e}")
            return image
    
    def super_resolution_enhance(self, image):
        """Apply super-resolution-like enhancement"""
        try:
            img_np = np.array(image)
            
            # 1. Edge-aware upsampling simulation
            # Even though we're not changing size, we can enhance as if upsampled
            print(f"[{VERSION}] Applying super-resolution enhancement...")
            
            # Create multiple shifted versions
            shifts = [(0,0), (1,0), (0,1), (1,1)]
            enhanced = np.zeros_like(img_np, dtype=np.float32)
            
            for dx, dy in shifts:
                shifted = np.roll(np.roll(img_np, dx, axis=1), dy, axis=0)
                enhanced += shifted.astype(np.float32)
            
            enhanced /= len(shifts)
            
            # 2. Edge enhancement using Laplacian
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian = np.abs(laplacian)
            laplacian = cv2.GaussianBlur(laplacian, (3, 3), 0)
            laplacian = np.clip(laplacian * 2, 0, 50)
            
            # Add edge details back
            for i in range(3):
                enhanced[:, :, i] += laplacian
            
            # 3. Frequency domain enhancement
            # Enhance high frequencies
            for i in range(3):
                channel = enhanced[:, :, i]
                blur = cv2.GaussianBlur(channel, (5, 5), 0)
                detail = channel - blur
                enhanced[:, :, i] = channel + detail * 0.5
            
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
            
            # 4. Final sharpening
            enhanced = Image.fromarray(enhanced)
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(1.3)
            
            return enhanced
            
        except Exception as e:
            print(f"[{VERSION}] Error in super-resolution: {e}")
            return image
    
    def create_perfect_thumbnail_1000x1300(self, image):
        """Create perfect 1000x1300 thumbnail with strong detail enhancement"""
        try:
            target_size = (1000, 1300)
            img_np = np.array(image)
            h, w = img_np.shape[:2]
            
            print(f"[{VERSION}] Creating detail-enhanced thumbnail...")
            
            # Simple center crop approach (proven effective in v33)
            crop_ratio = 1.3
            
            if w / h > 1 / crop_ratio:
                crop_h = int(h * 0.5)  # Use 50% of height
                crop_w = int(crop_h / crop_ratio)
            else:
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
            
            # Apply super-resolution-like enhancement
            final = self.super_resolution_enhance(final)
            
            print(f"[{VERSION}] Created 1000x1300 detail-enhanced thumbnail")
            return final
            
        except Exception as e:
            print(f"[{VERSION}] Error creating thumbnail: {e}")
            traceback.print_exc()
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
    """RunPod handler - V34 with strong detail enhancement"""
    print(f"[{VERSION}] ====== Thumbnail Handler Started ======")
    print(f"[{VERSION}] Strong Detail Enhancement & Noise/Dust/Scratch Removal")
    
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
        processor = ThumbnailProcessorV34()
        
        # Process image step by step
        had_black_frame = False
        
        # 1. Detect and remove black box
        try:
            image, had_black_frame = processor.detect_and_remove_black_box(image)
            print(f"[{VERSION}] Black box detection complete: {had_black_frame}")
        except Exception as e:
            print(f"[{VERSION}] Error in black frame detection: {e}")
            traceback.print_exc()
        
        # 2. Apply strong detail enhancement and cleanup - AFTER black box removal
        try:
            image = processor.apply_enhancement_matching_v34(image)
            print(f"[{VERSION}] Detail enhancement and cleanup applied")
        except Exception as e:
            print(f"[{VERSION}] Error in enhancement: {e}")
            traceback.print_exc()
        
        # 3. Create PERFECT 1000x1300 thumbnail with detail enhancement
        try:
            thumbnail = processor.create_perfect_thumbnail_1000x1300(image)
            print(f"[{VERSION}] Perfect detail-enhanced thumbnail created: {thumbnail.size}")
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
                "processing_method": "detail_enhanced_v34",
                "processing_time": round(processing_time, 2),
                "replicate_available": REPLICATE_AVAILABLE,
                "replicate_used": False,
                "enhancements_applied": [
                    "noise_removal",
                    "dust_scratch_removal", 
                    "detail_enhancement",
                    "super_resolution_simulation",
                    "micro_contrast",
                    "edge_sharpening"
                ],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                "warning": "Google Script must add padding: while (base64Data.length % 4 !== 0) { base64Data += '='; }"
            }
        }
        
        print(f"[{VERSION}] ====== Success - Returning Detail-Enhanced Thumbnail ======")
        print(f"[{VERSION}] Total processing time: {processing_time:.2f}s")
        print(f"[{VERSION}] Black frame detected and removed: {had_black_frame}")
        print(f"[{VERSION}] Detail enhancements applied successfully")
        
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
    print("V34 - Strong Detail Enhancement & Noise/Dust/Scratch Removal")
    print("Features:")
    print("- Simple black box detection and removal")
    print("- Advanced noise and defect removal")
    print("- Dust and scratch inpainting")
    print("- Super-resolution-like enhancement")
    print("- Micro-contrast and edge sharpening")
    print("- CLAHE for local contrast")
    print("- High-pass detail enhancement")
    print(f"Replicate Available: {REPLICATE_AVAILABLE}")
    print("CRITICAL: Padding is removed for Make.com")
    print("Google Apps Script MUST add padding back:")
    print("while (base64Data.length % 4 !== 0) { base64Data += '='; }")
    print("="*70)
    
    runpod.serverless.start({"handler": handler})
