import runpod
import base64
import numpy as np
from PIL import Image, ImageEnhance, ImageChops, ImageFilter
import cv2
import io
import os
import traceback
import time

# Version info
VERSION = "v28-thumbnail"

class ThumbnailProcessorV28:
    """v28 Thumbnail Processor - Stable Rectangle Detection with Error Handling"""
    
    def __init__(self):
        print(f"[{VERSION}] Initializing - Stable Rectangle Detection")
    
    def detect_black_rectangle_frame(self, image):
        """Detect black rectangle frame with better error handling"""
        try:
            img_np = np.array(image)
            h, w = img_np.shape[:2]
            print(f"[{VERSION}] Rectangle frame detection - Processing {w}x{h} image")
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
            # Method 1: Simple threshold-based detection
            # This is most reliable for clear black frames
            best_crop = None
            
            for thresh_val in [20, 30, 40, 50, 60, 70, 80]:
                # Find non-black pixels
                non_black = gray > thresh_val
                
                # Find bounding box of non-black content
                rows = np.any(non_black, axis=1)
                cols = np.any(non_black, axis=0)
                
                if np.any(rows) and np.any(cols):
                    try:
                        row_indices = np.where(rows)[0]
                        col_indices = np.where(cols)[0]
                        
                        if len(row_indices) > 0 and len(col_indices) > 0:
                            ymin, ymax = row_indices[0], row_indices[-1]
                            xmin, xmax = col_indices[0], col_indices[-1]
                            
                            # Validate crop dimensions
                            if xmax > xmin and ymax > ymin:
                                area = (xmax - xmin) * (ymax - ymin)
                                original_area = w * h
                                
                                # The crop should be significantly smaller than original
                                # but not too small
                                if 0.1 * original_area < area < 0.9 * original_area:
                                    best_crop = (xmin, ymin, xmax, ymax)
                                    print(f"[{VERSION}] Threshold {thresh_val}: Found valid crop")
                                    break
                    except Exception as e:
                        print(f"[{VERSION}] Error in threshold {thresh_val}: {e}")
                        continue
            
            # Method 2: Edge-based detection if threshold method fails
            if best_crop is None:
                try:
                    print(f"[{VERSION}] Trying edge-based detection")
                    
                    # Use Canny edge detection
                    edges = cv2.Canny(gray, 30, 100)
                    
                    # Find contours
                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        # Get bounding rect of all contours
                        all_points = []
                        for contour in contours:
                            if cv2.contourArea(contour) > 100:  # Filter small contours
                                all_points.extend(contour.reshape(-1, 2))
                        
                        if all_points:
                            all_points = np.array(all_points)
                            x_coords = all_points[:, 0]
                            y_coords = all_points[:, 1]
                            
                            xmin, xmax = np.min(x_coords), np.max(x_coords)
                            ymin, ymax = np.min(y_coords), np.max(y_coords)
                            
                            if xmax > xmin and ymax > ymin:
                                best_crop = (xmin, ymin, xmax, ymax)
                                print(f"[{VERSION}] Edge detection: Found crop")
                except Exception as e:
                    print(f"[{VERSION}] Error in edge detection: {e}")
            
            # Apply the best crop found
            if best_crop:
                xmin, ymin, xmax, ymax = best_crop
                
                # Add small margin
                margin = 5
                xmin = max(0, xmin - margin)
                ymin = max(0, ymin - margin)
                xmax = min(w, xmax + margin)
                ymax = min(h, ymax + margin)
                
                print(f"[{VERSION}] Final crop: ({xmin},{ymin}) to ({xmax},{ymax})")
                
                # Crop the image
                cropped = img_np[ymin:ymax, xmin:xmax]
                
                # Validate cropped image
                if cropped.size > 0:
                    return Image.fromarray(cropped), True
                else:
                    print(f"[{VERSION}] Invalid crop dimensions, returning original")
                    return image, False
            
            print(f"[{VERSION}] No black rectangle frame detected")
            return image, False
            
        except Exception as e:
            print(f"[{VERSION}] Error in black frame detection: {e}")
            traceback.print_exc()
            return image, False
    
    def apply_simple_enhancement(self, image):
        """Enhancement matching v28 enhancement handler"""
        try:
            # 1. Sharpening first
            image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=50, threshold=3))
            
            # 2. Brightness
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.22)  # Match v28
            
            # 3. Contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.15)  # Match v28
            
            # 4. Saturation
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(0.97)  # Match v28
            
            # 5. Convert to numpy for advanced processing
            img_np = np.array(image)
            h, w = img_np.shape[:2]
            
            # 6. Apply CLAHE for better brightness distribution
            img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(img_lab)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l_channel = clahe.apply(l_channel)
            
            img_lab = cv2.merge([l_channel, a_channel, b_channel])
            img_np = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
            
            # 7. Clean white background
            background_color = (250, 249, 247)
            
            # Edge-aware mask
            edges = cv2.Canny(cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY), 50, 150)
            edges_dilated = cv2.dilate(edges, np.ones((5,5), np.uint8), iterations=2)
            
            mask = np.ones((h, w), dtype=np.float32)
            mask[edges_dilated > 0] = 0
            mask = cv2.GaussianBlur(mask, (31, 31), 15)
            
            for i in range(3):
                img_np[:, :, i] = img_np[:, :, i] * (1 - mask * 0.1) + background_color[i] * mask * 0.1
            
            # 8. Sigmoid brightness adjustment
            img_float = img_np.astype(np.float32) / 255.0
            img_float = 1 / (1 + np.exp(-12 * (img_float - 0.45)))
            img_np = (img_float * 255).astype(np.uint8)
            
            return Image.fromarray(img_np)
            
        except Exception as e:
            print(f"[{VERSION}] Error in enhancement: {e}")
            return image
    
    def create_thumbnail_1000x1300(self, image):
        """Create exact 1000x1300 thumbnail with error handling"""
        try:
            target_size = (1000, 1300)
            
            # Find wedding ring area
            img_np = np.array(image)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
            # Find bright areas (rings) and edges
            _, bright = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
            edges = cv2.Canny(gray, 30, 80)
            combined = cv2.bitwise_or(bright, edges)
            
            # Clean up
            kernel = np.ones((5, 5), np.uint8)
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            cropped = None
            if contours:
                # Filter valid contours
                valid_contours = [c for c in contours if cv2.contourArea(c) > 200]
                
                if valid_contours:
                    # Get bounding box
                    all_points = np.concatenate(valid_contours)
                    x, y, w_box, h_box = cv2.boundingRect(all_points)
                    
                    # Tight padding (5%)
                    padding_x = int(w_box * 0.05)
                    padding_y = int(h_box * 0.05)
                    
                    x = max(0, x - padding_x)
                    y = max(0, y - padding_y)
                    w_box = min(img_np.shape[1] - x, w_box + 2 * padding_x)
                    h_box = min(img_np.shape[0] - y, h_box + 2 * padding_y)
                    
                    if w_box > 0 and h_box > 0:
                        cropped = image.crop((x, y, x + w_box, y + h_box))
            
            # Fallback if no valid crop found
            if cropped is None:
                # Center 60% crop
                w, h = image.size
                margin_x = int(w * 0.2)
                margin_y = int(h * 0.2)
                cropped = image.crop((margin_x, margin_y, w - margin_x, h - margin_y))
            
            # Resize to fill target size
            cropped_w, cropped_h = cropped.size
            scale = max(target_size[0] / cropped_w, target_size[1] / cropped_h)
            
            new_w = int(cropped_w * scale)
            new_h = int(cropped_h * scale)
            
            # Use high quality resampling
            resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Create final canvas
            canvas = Image.new('RGB', target_size, (250, 249, 247))
            
            # Center crop or paste
            if new_w > target_size[0] or new_h > target_size[1]:
                # Center crop
                left = (new_w - target_size[0]) // 2
                top = (new_h - target_size[1]) // 2
                resized = resized.crop((left, top, left + target_size[0], top + target_size[1]))
                canvas = resized
            else:
                # Center paste
                paste_x = (target_size[0] - new_w) // 2
                paste_y = (target_size[1] - new_h) // 2
                canvas.paste(resized, (paste_x, paste_y))
            
            # Final sharpening
            enhancer = ImageEnhance.Sharpness(canvas)
            canvas = enhancer.enhance(1.5)
            
            print(f"[{VERSION}] Created exact 1000x1300 thumbnail")
            return canvas
            
        except Exception as e:
            print(f"[{VERSION}] Error creating thumbnail: {e}")
            # Return a basic resize as fallback
            return image.resize(target_size, Image.Resampling.LANCZOS)

def handler(job):
    """RunPod handler - V28 with complete error handling"""
    print(f"[{VERSION}] ====== Thumbnail Handler Started ======")
    print(f"[{VERSION}] Stable Rectangle Detection with Error Handling")
    
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
            return {
                "output": {
                    "thumbnail": None,
                    "error": f"Failed to decode image: {str(e)}",
                    "success": False,
                    "version": VERSION
                }
            }
        
        # Create processor
        processor = ThumbnailProcessorV28()
        
        # Process image step by step with error handling
        had_black_frame = False
        
        # 1. RECTANGLE frame detection
        try:
            image, had_black_frame = processor.detect_black_rectangle_frame(image)
            print(f"[{VERSION}] Black frame detection: {had_black_frame}")
        except Exception as e:
            print(f"[{VERSION}] Error in black frame detection: {e}")
        
        # 2. Apply color enhancement
        try:
            image = processor.apply_simple_enhancement(image)
            print(f"[{VERSION}] Color enhancement applied")
        except Exception as e:
            print(f"[{VERSION}] Error in color enhancement: {e}")
        
        # 3. Create thumbnail
        try:
            thumbnail = processor.create_thumbnail_1000x1300(image)
            print(f"[{VERSION}] Thumbnail created: {thumbnail.size}")
        except Exception as e:
            print(f"[{VERSION}] Error creating thumbnail: {e}")
            # Fallback to simple resize
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
                "processing_method": "stable_rectangle_detection_v28",
                "processing_time": round(processing_time, 2),
                "warning": "Google Script must add padding: while (base64Data.length % 4 !== 0) { base64Data += '='; }"
            }
        }
        
        print(f"[{VERSION}] ====== Success - Returning Thumbnail ======")
        print(f"[{VERSION}] Total processing time: {processing_time:.2f}s")
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
    print("V28 - Stable Rectangle Detection with Error Handling")
    print("Features:")
    print("- Multiple threshold detection")
    print("- Edge-based fallback detection")
    print("- Complete error handling")
    print("- Matches v28 enhancement settings")
    print("CRITICAL: Padding is removed for Make.com")
    print("Google Apps Script MUST add padding back:")
    print("while (base64Data.length % 4 !== 0) { base64Data += '='; }")
    print("="*70)
    
    runpod.serverless.start({"handler": handler})
