import runpod
import base64
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import io
import os
import traceback
import time

# Version info
VERSION = "v29-thumbnail"

class ThumbnailProcessorV29:
    """v29 Thumbnail Processor - Perfect Rectangle Detection"""
    
    def __init__(self):
        print(f"[{VERSION}] Initializing - Perfect Rectangle Detection")
    
    def detect_black_rectangle_perfectly(self, image):
        """Detect black rectangle frame with 4-line detection method"""
        try:
            img_np = np.array(image)
            h, w = img_np.shape[:2]
            print(f"[{VERSION}] Perfect rectangle detection - Processing {w}x{h} image")
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
            # Method 1: Find 4 lines forming rectangle
            # Apply strong bilateral filter to reduce noise while keeping edges
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Strong edge detection
            edges = cv2.Canny(filtered, 30, 90)
            
            # Dilate edges to connect broken lines
            kernel = np.ones((3,3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            
            # Detect lines using HoughLinesP
            lines = cv2.HoughLinesP(edges, 
                                   rho=1, 
                                   theta=np.pi/180, 
                                   threshold=100,
                                   minLineLength=min(w, h) * 0.2,  # At least 20% of image size
                                   maxLineGap=20)
            
            if lines is not None and len(lines) >= 4:
                # Separate horizontal and vertical lines
                h_lines = []
                v_lines = []
                
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    
                    # Calculate line angle
                    angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                    
                    # Horizontal lines (angle close to 0 or 180)
                    if angle < 10 or angle > 170:
                        h_lines.append(((x1 + x2) // 2, (y1 + y2) // 2, abs(x2 - x1)))
                    # Vertical lines (angle close to 90)
                    elif 80 < angle < 100:
                        v_lines.append(((x1 + x2) // 2, (y1 + y2) // 2, abs(y2 - y1)))
                
                # Find rectangle bounds from lines
                if len(h_lines) >= 2 and len(v_lines) >= 2:
                    # Sort lines by position
                    h_lines.sort(key=lambda x: x[1])  # Sort by y coordinate
                    v_lines.sort(key=lambda x: x[0])  # Sort by x coordinate
                    
                    # Find outermost lines that could form a rectangle
                    top_line = None
                    bottom_line = None
                    left_line = None
                    right_line = None
                    
                    # Find top and bottom horizontal lines
                    for line in h_lines:
                        if line[1] < h * 0.4:  # Top 40% of image
                            top_line = line[1]
                        elif line[1] > h * 0.6:  # Bottom 40% of image
                            if bottom_line is None or line[1] > bottom_line:
                                bottom_line = line[1]
                    
                    # Find left and right vertical lines
                    for line in v_lines:
                        if line[0] < w * 0.4:  # Left 40% of image
                            left_line = line[0]
                        elif line[0] > w * 0.6:  # Right 40% of image
                            if right_line is None or line[0] > right_line:
                                right_line = line[0]
                    
                    # If we found all 4 lines, we have a rectangle
                    if all([top_line, bottom_line, left_line, right_line]):
                        print(f"[{VERSION}] Found rectangle from lines: T:{top_line}, B:{bottom_line}, L:{left_line}, R:{right_line}")
                        
                        # Add small margin inside the black frame
                        margin = 20
                        crop_box = (
                            int(left_line + margin),
                            int(top_line + margin),
                            int(right_line - margin),
                            int(bottom_line - margin)
                        )
                        
                        # Validate crop box
                        if crop_box[2] > crop_box[0] and crop_box[3] > crop_box[1]:
                            cropped = img_np[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
                            return Image.fromarray(cropped), True
            
            # Method 2: Threshold-based detection for solid black rectangles
            # Try multiple thresholds
            for thresh_val in [20, 30, 40, 50]:
                _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
                
                # Find contours
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Sort by area
                    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
                    
                    for contour in sorted_contours[:3]:
                        # Get bounding rectangle
                        x, y, cw, ch = cv2.boundingRect(contour)
                        
                        # Check if it's a significant rectangle
                        area = cw * ch
                        if area > (w * h * 0.1):  # At least 10% of image
                            # Approximate contour to polygon
                            epsilon = 0.02 * cv2.arcLength(contour, True)
                            approx = cv2.approxPolyDP(contour, epsilon, True)
                            
                            # Check if it's a rectangle (4 vertices)
                            if len(approx) == 4:
                                # Check aspect ratio (roughly square for wedding ring boxes)
                                aspect_ratio = cw / ch
                                if 0.7 < aspect_ratio < 1.3:
                                    print(f"[{VERSION}] Found rectangle contour at ({x},{y}) size {cw}x{ch}")
                                    
                                    # Check if center is brighter (hollow rectangle)
                                    center_region = gray[y+ch//4:y+3*ch//4, x+cw//4:x+3*cw//4]
                                    edge_region = gray[y:y+20, x:x+cw]
                                    
                                    if np.mean(center_region) > np.mean(edge_region) + 30:
                                        # This is our black frame!
                                        # Crop inside the frame
                                        margin = 20
                                        crop_box = (
                                            max(0, x + margin),
                                            max(0, y + margin),
                                            min(w, x + cw - margin),
                                            min(h, y + ch - margin)
                                        )
                                        
                                        cropped = img_np[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
                                        return Image.fromarray(cropped), True
            
            # Method 3: Scan from edges to find frame boundaries
            print(f"[{VERSION}] Trying edge scanning method")
            
            # Find where black frame starts/ends
            threshold = 50
            
            # Scan from top
            top_boundary = 0
            for y in range(h//3):
                row_mean = np.mean(gray[y, :])
                if row_mean < threshold:
                    top_boundary = y
                elif top_boundary > 0 and row_mean > threshold + 50:
                    # Found transition from black to white
                    top_boundary = y
                    break
            
            # Scan from bottom
            bottom_boundary = h
            for y in range(h//3):
                row_mean = np.mean(gray[h-1-y, :])
                if row_mean < threshold:
                    bottom_boundary = h - y
                elif bottom_boundary < h and row_mean > threshold + 50:
                    bottom_boundary = h - y
                    break
            
            # Scan from left
            left_boundary = 0
            for x in range(w//3):
                col_mean = np.mean(gray[:, x])
                if col_mean < threshold:
                    left_boundary = x
                elif left_boundary > 0 and col_mean > threshold + 50:
                    left_boundary = x
                    break
            
            # Scan from right
            right_boundary = w
            for x in range(w//3):
                col_mean = np.mean(gray[:, w-1-x])
                if col_mean < threshold:
                    right_boundary = w - x
                elif right_boundary < w and col_mean > threshold + 50:
                    right_boundary = w - x
                    break
            
            # Check if we found a significant frame
            if (top_boundary > 20 or bottom_boundary < h - 20 or 
                left_boundary > 20 or right_boundary < w - 20):
                
                print(f"[{VERSION}] Edge scan found frame: T:{top_boundary}, B:{bottom_boundary}, L:{left_boundary}, R:{right_boundary}")
                
                # Crop with margin
                margin = 10
                crop_box = (
                    left_boundary + margin,
                    top_boundary + margin,
                    right_boundary - margin,
                    bottom_boundary - margin
                )
                
                if crop_box[2] > crop_box[0] and crop_box[3] > crop_box[1]:
                    cropped = img_np[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
                    return Image.fromarray(cropped), True
            
            print(f"[{VERSION}] No black rectangle frame detected")
            return image, False
            
        except Exception as e:
            print(f"[{VERSION}] Error in black frame detection: {e}")
            traceback.print_exc()
            return image, False
    
    def apply_enhancement_matching_v29(self, image):
        """Enhancement matching v29 enhancement handler"""
        try:
            # Match v29 enhancement settings
            # 1. Pre-sharpening
            image = image.filter(ImageFilter.UnsharpMask(radius=1.5, percent=60, threshold=2))
            
            # 2. Brightness
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.28)  # Match v29
            
            # 3. Contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.18)  # Match v29
            
            # 4. Saturation
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(0.92)  # Match v29
            
            # 5. Convert to numpy for advanced processing
            img_np = np.array(image)
            h, w = img_np.shape[:2]
            
            # 6. Apply CLAHE with v29 settings
            img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(img_lab)
            
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l_channel = clahe.apply(l_channel)
            l_channel = cv2.add(l_channel, 15)
            l_channel = np.clip(l_channel, 0, 255)
            
            img_lab = cv2.merge([l_channel, a_channel, b_channel])
            img_np = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
            
            # 7. Pure white background
            background_color = (253, 252, 251)
            white_overlay = np.full((h, w, 3), background_color, dtype=np.float32)
            
            # Edge detection
            edges = cv2.Canny(cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY), 40, 120)
            edges_dilated = cv2.dilate(edges, np.ones((7,7), np.uint8), iterations=2)
            
            mask = np.ones((h, w), dtype=np.float32)
            mask[edges_dilated > 0] = 0
            mask = cv2.GaussianBlur(mask, (51, 51), 25)
            
            for i in range(3):
                img_np[:, :, i] = img_np[:, :, i] * (1 - mask * 0.25) + white_overlay[:, :, i] * mask * 0.25
            
            # 8. Gamma correction
            gamma = 0.85
            img_np = np.power(img_np / 255.0, gamma) * 255
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            
            return Image.fromarray(img_np)
            
        except Exception as e:
            print(f"[{VERSION}] Error in enhancement: {e}")
            return image
    
    def create_perfect_thumbnail_1000x1300(self, image):
        """Create perfect 1000x1300 thumbnail with ring focus"""
        try:
            target_size = (1000, 1300)
            
            # Find wedding ring area using multiple methods
            img_np = np.array(image)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
            # Method 1: Find bright metallic areas (rings)
            # Rings are usually bright and have high local contrast
            # Apply adaptive threshold
            adaptive_thresh = cv2.adaptiveThreshold(gray, 255, 
                                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                  cv2.THRESH_BINARY, 11, 2)
            
            # Find edges of rings
            edges = cv2.Canny(gray, 50, 150)
            
            # Combine methods
            combined = cv2.bitwise_or(adaptive_thresh, edges)
            
            # Clean up with morphology
            kernel = np.ones((5, 5), np.uint8)
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
            combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            ring_box = None
            if contours:
                # Filter contours by area and circularity
                valid_contours = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 500:  # Minimum area for a ring
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
                    
                    # Very tight padding for maximum ring size
                    padding_percent = 0.08  # Only 8% padding
                    padding_x = int(w_box * padding_percent)
                    padding_y = int(h_box * padding_percent)
                    
                    x = max(0, x - padding_x)
                    y = max(0, y - padding_y)
                    w_box = min(img_np.shape[1] - x, w_box + 2 * padding_x)
                    h_box = min(img_np.shape[0] - y, h_box + 2 * padding_y)
                    
                    ring_box = (x, y, w_box, h_box)
            
            # If no rings found, use center crop
            if ring_box is None:
                h, w = image.size[1], image.size[0]
                # Assume rings are in center 50%
                margin_x = int(w * 0.25)
                margin_y = int(h * 0.25)
                ring_box = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)
            
            # Crop to ring area
            x, y, w_box, h_box = ring_box
            cropped = image.crop((x, y, x + w_box, y + h_box))
            
            # Calculate scale to fill 1000x1300
            cropped_w, cropped_h = cropped.size
            scale_w = target_size[0] / cropped_w
            scale_h = target_size[1] / cropped_h
            scale = max(scale_w, scale_h)  # Fill the entire target
            
            # Apply scale
            new_w = int(cropped_w * scale)
            new_h = int(cropped_h * scale)
            
            # High quality resize
            resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Create final 1000x1300 image
            if new_w > target_size[0] or new_h > target_size[1]:
                # Center crop to exact size
                left = (new_w - target_size[0]) // 2
                top = (new_h - target_size[1]) // 2
                final = resized.crop((left, top, left + target_size[0], top + target_size[1]))
            else:
                # Should not happen with max scale, but just in case
                final = Image.new('RGB', target_size, (253, 252, 251))
                paste_x = (target_size[0] - new_w) // 2
                paste_y = (target_size[1] - new_h) // 2
                final.paste(resized, (paste_x, paste_y))
            
            # Final detail enhancement
            # Strong sharpening for product detail
            enhancer = ImageEnhance.Sharpness(final)
            final = enhancer.enhance(1.8)  # Strong sharpening
            
            # Additional edge enhancement
            final_np = np.array(final)
            
            # Unsharp mask for extra detail
            gaussian = cv2.GaussianBlur(final_np, (0, 0), 2.0)
            unsharp = cv2.addWeighted(final_np, 1.5, gaussian, -0.5, 0)
            
            # Ensure no over-sharpening artifacts
            unsharp = np.clip(unsharp, 0, 255).astype(np.uint8)
            
            print(f"[{VERSION}] Created perfect 1000x1300 thumbnail")
            return Image.fromarray(unsharp)
            
        except Exception as e:
            print(f"[{VERSION}] Error creating thumbnail: {e}")
            # Fallback to simple resize
            return image.resize((1000, 1300), Image.Resampling.LANCZOS)

def handler(job):
    """RunPod handler - V29 with perfect rectangle detection"""
    print(f"[{VERSION}] ====== Thumbnail Handler Started ======")
    print(f"[{VERSION}] Perfect Rectangle Detection & Processing")
    
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
        processor = ThumbnailProcessorV29()
        
        # Process image step by step
        had_black_frame = False
        
        # 1. PERFECT RECTANGLE frame detection
        try:
            image, had_black_frame = processor.detect_black_rectangle_perfectly(image)
            print(f"[{VERSION}] Black rectangle detection: {had_black_frame}")
        except Exception as e:
            print(f"[{VERSION}] Error in black frame detection: {e}")
        
        # 2. Apply color enhancement matching v29
        try:
            image = processor.apply_enhancement_matching_v29(image)
            print(f"[{VERSION}] Color enhancement applied (v29 style)")
        except Exception as e:
            print(f"[{VERSION}] Error in color enhancement: {e}")
        
        # 3. Create perfect 1000x1300 thumbnail
        try:
            thumbnail = processor.create_perfect_thumbnail_1000x1300(image)
            print(f"[{VERSION}] Perfect thumbnail created: {thumbnail.size}")
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
                "processing_method": "perfect_rectangle_detection_v29",
                "processing_time": round(processing_time, 2),
                "warning": "Google Script must add padding: while (base64Data.length % 4 !== 0) { base64Data += '='; }"
            }
        }
        
        print(f"[{VERSION}] ====== Success - Returning Perfect Thumbnail ======")
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
    print("V29 - Perfect Rectangle Detection")
    print("Features:")
    print("- 4-line rectangle detection")
    print("- Multiple threshold checking")
    print("- Edge scanning with transitions")
    print("- Ring-focused cropping")
    print("- Strong detail enhancement")
    print("- Matches v29 enhancement settings")
    print("CRITICAL: Padding is removed for Make.com")
    print("Google Apps Script MUST add padding back:")
    print("while (base64Data.length % 4 !== 0) { base64Data += '='; }")
    print("="*70)
    
    runpod.serverless.start({"handler": handler})
