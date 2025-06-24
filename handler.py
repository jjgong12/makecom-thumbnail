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
VERSION = "v32-thumbnail"

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

class ThumbnailProcessorV32:
    """v32 Thumbnail Processor - Bright Thumbnail with Less Zoom"""
    
    def __init__(self):
        print(f"[{VERSION}] Initializing - Bright Thumbnail with Less Zoom")
        self.replicate_client = None
    
    def detect_black_box_ultimate(self, image):
        """Ultimate black box detection with multiple advanced methods"""
        try:
            img_np = np.array(image)
            h, w = img_np.shape[:2]
            print(f"[{VERSION}] Ultimate black box detection - Processing {w}x{h} image")
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
            # Method 1: Ultra-sensitive black detection
            print(f"[{VERSION}] Method 1: Ultra-sensitive black detection")
            
            # Very low threshold for black
            black_threshold = 25
            
            # Create mask of black pixels
            black_mask = gray < black_threshold
            
            # Find largest black rectangle
            # Sum black pixels in each row and column
            row_sums = np.sum(black_mask, axis=1)
            col_sums = np.sum(black_mask, axis=0)
            
            # Find continuous black regions
            # Top edge
            top = 0
            for i in range(h//2):
                if row_sums[i] > w * 0.7:  # 70% of row is black
                    top = i
                else:
                    if i > 10:
                        top = i
                        break
            
            # Bottom edge
            bottom = h
            for i in range(h//2):
                if row_sums[h-1-i] > w * 0.7:
                    bottom = h - i
                else:
                    if i > 10:
                        bottom = h - i
                        break
            
            # Left edge
            left = 0
            for i in range(w//2):
                if col_sums[i] > h * 0.7:
                    left = i
                else:
                    if i > 10:
                        left = i
                        break
            
            # Right edge
            right = w
            for i in range(w//2):
                if col_sums[w-1-i] > h * 0.7:
                    right = w - i
                else:
                    if i > 10:
                        right = w - i
                        break
            
            print(f"[{VERSION}] Initial detection - T:{top}, B:{bottom}, L:{left}, R:{right}")
            
            # Check if we found a significant black frame
            if (top > 20 or (h - bottom) > 20 or left > 20 or (w - right) > 20):
                print(f"[{VERSION}] Black frame detected by edge analysis!")
                
                # Add margin to remove all black
                margin = 30
                crop_box = (
                    left + margin,
                    top + margin,
                    right - margin,
                    bottom - margin
                )
                
                if crop_box[2] > crop_box[0] and crop_box[3] > crop_box[1]:
                    cropped = img_np[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
                    return Image.fromarray(cropped), True
            
            # Method 2: Find black rectangles using contours
            print(f"[{VERSION}] Method 2: Contour-based black rectangle detection")
            
            # Multiple thresholds
            for thresh in [20, 30, 40, 50]:
                _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
                
                # Clean up
                kernel = np.ones((10, 10), np.uint8)
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                
                # Find contours
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Sort by area
                    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
                    
                    for contour in sorted_contours[:3]:
                        x, y, cw, ch = cv2.boundingRect(contour)
                        area = cw * ch
                        
                        # Check if it's a significant black area
                        if area > (w * h * 0.05):  # At least 5% of image
                            # Check if it's rectangular
                            rect_area = cv2.minAreaRect(contour)
                            box = cv2.boxPoints(rect_area)
                            box = np.int0(box)
                            
                            # Calculate rectangularity
                            contour_area = cv2.contourArea(contour)
                            rect_area_calc = cw * ch
                            rectangularity = contour_area / rect_area_calc if rect_area_calc > 0 else 0
                            
                            if rectangularity > 0.8:  # It's quite rectangular
                                print(f"[{VERSION}] Black rectangle found at ({x},{y}) size {cw}x{ch}")
                                
                                # Check if content is inside (brighter center)
                                center_region = gray[y+ch//4:y+3*ch//4, x+cw//4:x+3*cw//4]
                                
                                if center_region.size > 0 and np.mean(center_region) > thresh + 20:
                                    # Content is inside the black box
                                    margin = 30
                                    crop_x1 = x + margin
                                    crop_y1 = y + margin
                                    crop_x2 = x + cw - margin
                                    crop_y2 = y + ch - margin
                                    
                                    if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                                        cropped = img_np[crop_y1:crop_y2, crop_x1:crop_x2]
                                        return Image.fromarray(cropped), True
            
            # Method 3: Advanced scanning from all directions
            print(f"[{VERSION}] Method 3: Advanced directional scanning")
            
            # Scan multiple lines from each edge
            scan_lines = 10
            black_threshold = 30
            
            # Top scan
            top_values = []
            for line in range(scan_lines):
                x_pos = w // (scan_lines + 1) * (line + 1)
                for y in range(h//2):
                    if gray[y, x_pos] > black_threshold:
                        top_values.append(y)
                        break
            
            # Bottom scan
            bottom_values = []
            for line in range(scan_lines):
                x_pos = w // (scan_lines + 1) * (line + 1)
                for y in range(h//2):
                    if gray[h-1-y, x_pos] > black_threshold:
                        bottom_values.append(h - y)
                        break
            
            # Left scan
            left_values = []
            for line in range(scan_lines):
                y_pos = h // (scan_lines + 1) * (line + 1)
                for x in range(w//2):
                    if gray[y_pos, x] > black_threshold:
                        left_values.append(x)
                        break
            
            # Right scan
            right_values = []
            for line in range(scan_lines):
                y_pos = h // (scan_lines + 1) * (line + 1)
                for x in range(w//2):
                    if gray[y_pos, w-1-x] > black_threshold:
                        right_values.append(w - x)
                        break
            
            if top_values and bottom_values and left_values and right_values:
                # Use median values for robustness
                top = int(np.median(top_values))
                bottom = int(np.median(bottom_values))
                left = int(np.median(left_values))
                right = int(np.median(right_values))
                
                print(f"[{VERSION}] Multi-line scan - T:{top}, B:{bottom}, L:{left}, R:{right}")
                
                if (top > 20 or (h - bottom) > 20 or left > 20 or (w - right) > 20):
                    margin = 20
                    crop_box = (
                        left + margin,
                        top + margin,
                        right - margin,
                        bottom - margin
                    )
                    
                    if crop_box[2] > crop_box[0] and crop_box[3] > crop_box[1]:
                        cropped = img_np[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
                        return Image.fromarray(cropped), True
            
            # Method 4: Histogram-based detection
            print(f"[{VERSION}] Method 4: Histogram analysis")
            
            # Analyze color distribution
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            
            # Check for bimodal distribution (black frame + content)
            black_peak = np.sum(hist[:30])  # Very dark pixels
            total_pixels = h * w
            black_ratio = black_peak / total_pixels
            
            print(f"[{VERSION}] Black pixel ratio: {black_ratio:.2f}")
            
            if black_ratio > 0.1:  # More than 10% black pixels
                # Use Otsu's method to find optimal threshold
                _, otsu_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Find the content area (white in binary)
                contours, _ = cv2.findContours(otsu_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    x, y, cw, ch = cv2.boundingRect(largest)
                    
                    # Add small margin
                    margin = 20
                    x = max(0, x - margin)
                    y = max(0, y - margin)
                    cw = min(w - x, cw + 2 * margin)
                    ch = min(h - y, ch + 2 * margin)
                    
                    if cw < w * 0.9 and ch < h * 0.9:  # Content is smaller than image
                        print(f"[{VERSION}] Content area found by Otsu: ({x},{y}) size {cw}x{ch}")
                        cropped = img_np[y:y+ch, x:x+cw]
                        return Image.fromarray(cropped), True
            
            print(f"[{VERSION}] No black box detected by any method")
            return image, False
            
        except Exception as e:
            print(f"[{VERSION}] Error in black frame detection: {e}")
            traceback.print_exc()
            return image, False
    
    def remove_black_frame_replicate(self, image, had_frame):
        """Use Replicate API for inpainting if needed"""
        if not had_frame or not REPLICATE_AVAILABLE:
            return image
        
        try:
            print(f"[{VERSION}] Using Replicate for additional cleanup")
            
            # Initialize client
            if not self.replicate_client:
                api_token = os.environ.get('REPLICATE_API_TOKEN')
                if api_token:
                    self.replicate_client = replicate.Client(api_token=api_token)
                else:
                    print(f"[{VERSION}] No REPLICATE_API_TOKEN found")
                    return image
            
            # Convert to base64
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Run inpainting to clean any remaining artifacts
            output = self.replicate_client.run(
                "stability-ai/stable-diffusion-inpainting",
                input={
                    "image": f"data:image/png;base64,{img_base64}",
                    "prompt": "clean white background, professional product photography",
                    "negative_prompt": "black edges, dark corners, shadows",
                    "num_inference_steps": 20
                }
            )
            
            if output and len(output) > 0:
                response = requests.get(output[0])
                return Image.open(io.BytesIO(response.content))
                
        except Exception as e:
            print(f"[{VERSION}] Replicate error: {e}")
        
        return image
    
    def apply_enhancement_matching_v32(self, image):
        """Enhancement matching v32 enhancement handler - minimal bright"""
        try:
            # Match v32 enhancement settings - very minimal
            # 1. Light sharpening
            image = image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=50, threshold=3))
            
            # 2. Minimal brightness
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.08)  # Match v32
            
            # 3. Slight contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.06)  # Match v32
            
            # 4. Natural colors
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(0.98)  # Match v32
            
            # 5. Convert to numpy for subtle processing
            img_np = np.array(image)
            h, w = img_np.shape[:2]
            
            # 6. Very subtle white background
            white_color = (248, 248, 248)
            
            # Edge detection
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 60, 150)
            edges_dilated = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
            
            mask = np.ones((h, w), dtype=np.float32)
            mask[edges_dilated > 0] = 0
            mask = cv2.GaussianBlur(mask, (31, 31), 15)
            
            for i in range(3):
                img_np[:, :, i] = img_np[:, :, i] * (1 - mask * 0.08) + white_color[i] * mask * 0.08
            
            # Gentle gamma correction
            gamma = 0.95
            img_np = np.power(img_np / 255.0, gamma) * 255
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            
            return Image.fromarray(img_np)
            
        except Exception as e:
            print(f"[{VERSION}] Error in enhancement: {e}")
            return image
    
    def create_perfect_thumbnail_1000x1300(self, image):
        """Create perfect 1000x1300 thumbnail with less zoom"""
        try:
            target_size = (1000, 1300)
            
            # Find wedding ring area
            img_np = np.array(image)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
            print(f"[{VERSION}] Finding rings for perfect thumbnail with less zoom...")
            
            # Method 1: Find bright/metallic areas (rings)
            # Apply CLAHE for better contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced_gray = clahe.apply(gray)
            
            # Find bright regions
            _, bright_mask = cv2.threshold(enhanced_gray, 150, 255, cv2.THRESH_BINARY)
            
            # Find edges
            edges = cv2.Canny(enhanced_gray, 30, 100)
            
            # Combine
            combined = cv2.bitwise_or(bright_mask, edges)
            
            # Clean up
            kernel = np.ones((5, 5), np.uint8)
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
            combined = cv2.dilate(combined, kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            ring_box = None
            if contours:
                # Filter for ring-like shapes
                valid_contours = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 500:  # Minimum area
                        # Check circularity
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                            if circularity > 0.3:  # Reasonably circular
                                valid_contours.append(contour)
                
                if valid_contours:
                    # Get bounding box of all rings
                    all_points = np.concatenate(valid_contours)
                    x, y, w_box, h_box = cv2.boundingRect(all_points)
                    
                    # MORE PADDING for less zoom - like image 5
                    padding_percent = 0.15  # 15% padding - much more than before
                    padding_x = int(w_box * padding_percent)
                    padding_y = int(h_box * padding_percent)
                    
                    x = max(0, x - padding_x)
                    y = max(0, y - padding_y)
                    w_box = min(img_np.shape[1] - x, w_box + 2 * padding_x)
                    h_box = min(img_np.shape[0] - y, h_box + 2 * padding_y)
                    
                    ring_box = (x, y, w_box, h_box)
                    print(f"[{VERSION}] Rings found at ({x},{y}) size {w_box}x{h_box} with 15% padding")
            
            # If no rings found, use center area with more margin
            if ring_box is None:
                print(f"[{VERSION}] Using center crop fallback with more margin")
                h, w = image.size[1], image.size[0]
                margin_percent = 0.25  # 25% margin for less zoom
                margin_x = int(w * margin_percent)
                margin_y = int(h * margin_percent)
                ring_box = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)
            
            # Crop to ring area
            x, y, w_box, h_box = ring_box
            cropped = image.crop((x, y, x + w_box, y + h_box))
            
            print(f"[{VERSION}] Cropped to ring area: {cropped.size}")
            
            # Calculate scale to fill thumbnail - less aggressive
            cropped_w, cropped_h = cropped.size
            scale_w = target_size[0] / cropped_w
            scale_h = target_size[1] / cropped_h
            scale = max(scale_w, scale_h) * 1.02  # Only 2% extra instead of 10%
            
            new_w = int(cropped_w * scale)
            new_h = int(cropped_h * scale)
            
            # High quality resize
            resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Center crop to exact size
            left = (new_w - target_size[0]) // 2
            top = (new_h - target_size[1]) // 2
            final = resized.crop((left, top, left + target_size[0], top + target_size[1]))
            
            # Moderate detail enhancement - not too strong
            # Sharpening for clarity
            enhancer = ImageEnhance.Sharpness(final)
            final = enhancer.enhance(1.4)  # Moderate sharpening
            
            # Additional detail enhancement
            final_np = np.array(final)
            
            # Unsharp mask
            gaussian = cv2.GaussianBlur(final_np, (0, 0), 2.0)
            unsharp = cv2.addWeighted(final_np, 1.4, gaussian, -0.4, 0)
            
            # Edge enhancement - less aggressive
            kernel = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]]) / 10.0  # Softer kernel
            
            enhanced = cv2.filter2D(unsharp, -1, kernel)
            final_np = cv2.addWeighted(unsharp, 0.8, enhanced, 0.2, 0)  # Less edge enhancement
            
            # Ensure no artifacts
            final_np = np.clip(final_np, 0, 255).astype(np.uint8)
            
            print(f"[{VERSION}] Created perfect 1000x1300 thumbnail with less zoom")
            return Image.fromarray(final_np)
            
        except Exception as e:
            print(f"[{VERSION}] Error creating thumbnail: {e}")
            traceback.print_exc()
            return image.resize((1000, 1300), Image.Resampling.LANCZOS)

def handler(job):
    """RunPod handler - V32 with bright processing and less zoom"""
    print(f"[{VERSION}] ====== Thumbnail Handler Started ======")
    print(f"[{VERSION}] Bright Processing & Less Zoom")
    
    start_time = time.time()
    
    try:
        job_input = job.get("input", {})
        print(f"[{VERSION}] Input type: {type(job_input)}")
        print(f"[{VERSION}] Input keys: {list(job_input.keys()) if isinstance(job_input, dict) else 'Not a dict'}")
        
        # Find base64 image
        base64_image = None
        
        if isinstance(job_input, dict):
            for key in ['image_base64', 'image', 'base64', 'data', 'input', 'file', 'imageData']:
                if key in job_input:
                    value = job_input[key]
                    if isinstance(value, str) and len(value) > 100:
                        base64_image = value
                        print(f"[{VERSION}] Found image in key: {key}")
                        break
        
        # Check nested structure
        if not base64_image and isinstance(job_input, dict):
            for key, value in job_input.items():
                if isinstance(value, dict):
                    for sub_key in ['image_base64', 'image', 'base64', 'data']:
                        if sub_key in value and isinstance(value[sub_key], str) and len(value[sub_key]) > 100:
                            base64_image = value[sub_key]
                            print(f"[{VERSION}] Found image in nested: {key}.{sub_key}")
                            break
                if base64_image:
                    break
        
        # Direct string input
        if not base64_image and isinstance(job_input, str) and len(job_input) > 100:
            base64_image = job_input
            print(f"[{VERSION}] Input was direct base64 string")
        
        if not base64_image:
            print(f"[{VERSION}] ERROR: No base64 image found!")
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
        
        # Process image
        print(f"[{VERSION}] Processing base64 image...")
        
        # Handle data URL format
        if ',' in base64_image and base64_image.startswith('data:'):
            base64_image = base64_image.split(',')[1]
            print(f"[{VERSION}] Removed data URL prefix")
        
        # Clean base64
        base64_image = base64_image.strip()
        
        # Add padding for decoding
        padding_needed = 4 - len(base64_image) % 4
        if padding_needed != 4:
            base64_image += '=' * padding_needed
            print(f"[{VERSION}] Added {padding_needed} padding characters for decoding")
        
        # Decode
        try:
            img_data = base64.b64decode(base64_image)
            image = Image.open(io.BytesIO(img_data))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            print(f"[{VERSION}] Image decoded: {image.size}, mode: {image.mode}")
        except Exception as e:
            print(f"[{VERSION}] DECODE ERROR: {e}")
            return {
                "output": {
                    "thumbnail": None,
                    "error": f"Failed to decode image: {str(e)}",
                    "success": False,
                    "version": VERSION
                }
            }
        
        # Create processor
        processor = ThumbnailProcessorV32()
        
        # Process image step by step
        had_black_frame = False
        
        # 1. ULTIMATE black box detection
        try:
            image, had_black_frame = processor.detect_black_box_ultimate(image)
            print(f"[{VERSION}] Black box detection complete: {had_black_frame}")
            
            # Additional cleanup with Replicate if available
            if had_black_frame and REPLICATE_AVAILABLE:
                image = processor.remove_black_frame_replicate(image, had_black_frame)
                print(f"[{VERSION}] Additional cleanup with Replicate done")
                
        except Exception as e:
            print(f"[{VERSION}] Error in black frame detection: {e}")
            traceback.print_exc()
        
        # 2. Apply color enhancement matching v32 - AFTER masking removal
        try:
            image = processor.apply_enhancement_matching_v32(image)
            print(f"[{VERSION}] Color enhancement applied (v32 style - minimal bright)")
        except Exception as e:
            print(f"[{VERSION}] Error in color enhancement: {e}")
            traceback.print_exc()
        
        # 3. Create PERFECT 1000x1300 thumbnail with LESS ZOOM
        try:
            thumbnail = processor.create_perfect_thumbnail_1000x1300(image)
            print(f"[{VERSION}] Perfect thumbnail created with less zoom: {thumbnail.size}")
        except Exception as e:
            print(f"[{VERSION}] Error creating thumbnail: {e}")
            traceback.print_exc()
            thumbnail = image.resize((1000, 1300), Image.Resampling.LANCZOS)
        
        # Convert to base64
        try:
            buffer = io.BytesIO()
            thumbnail.save(buffer, format='PNG', quality=95)
            buffer.seek(0)
            
            thumbnail_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # CRITICAL FOR MAKE.COM: Remove padding
            thumbnail_base64 = thumbnail_base64.rstrip('=')
            
            print(f"[{VERSION}] Thumbnail base64 length: {len(thumbnail_base64)}")
        except Exception as e:
            print(f"[{VERSION}] ENCODE ERROR: {e}")
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
                "processing_method": "bright_less_zoom_v32",
                "processing_time": round(processing_time, 2),
                "replicate_available": REPLICATE_AVAILABLE,
                "warning": "Google Script must add padding: while (base64Data.length % 4 !== 0) { base64Data += '='; }"
            }
        }
        
        print(f"[{VERSION}] ====== Success - Returning Bright Thumbnail with Less Zoom ======")
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
    print("V32 - Bright Thumbnail with Less Zoom")
    print("Features:")
    print("- Ultra-sensitive black detection (threshold: 25)")
    print("- Multi-line edge scanning")
    print("- Contour-based rectangle detection")
    print("- Histogram analysis")
    print("- Replicate API cleanup")
    print("- Less zoom (15% padding)")
    print("- Bright enhancement matching v32")
    print("- Moderate detail enhancement")
    print(f"Replicate Available: {REPLICATE_AVAILABLE}")
    print("CRITICAL: Padding is removed for Make.com")
    print("Google Apps Script MUST add padding back:")
    print("while (base64Data.length % 4 !== 0) { base64Data += '='; }")
    print("="*70)
    
    runpod.serverless.start({"handler": handler})
