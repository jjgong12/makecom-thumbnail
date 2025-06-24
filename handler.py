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
VERSION = "v30-thumbnail"

class ThumbnailProcessorV30:
    """v30 Thumbnail Processor - Perfect Crop & Detection"""
    
    def __init__(self):
        print(f"[{VERSION}] Initializing - Perfect Crop & Detection")
    
    def detect_black_rectangle_complete(self, image):
        """Complete black rectangle detection including bottom edges"""
        try:
            img_np = np.array(image)
            h, w = img_np.shape[:2]
            print(f"[{VERSION}] Complete rectangle detection - Processing {w}x{h} image")
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
            # Method 1: Aggressive edge scanning (especially for bottom)
            print(f"[{VERSION}] Scanning all edges carefully...")
            
            # Use lower threshold for better black detection
            threshold = 40
            
            # Scan from each edge with more precision
            # TOP edge
            top_boundary = 0
            for y in range(min(h//2, 300)):
                row = gray[y, :]
                black_pixels = np.sum(row < threshold)
                if black_pixels > w * 0.8:  # 80% of row is black
                    top_boundary = y
                else:
                    if y > 10:  # Found content
                        top_boundary = y
                        break
            
            # BOTTOM edge - scan more carefully
            bottom_boundary = h
            for y in range(min(h//2, 300)):
                row = gray[h-1-y, :]
                black_pixels = np.sum(row < threshold)
                if black_pixels > w * 0.8:  # 80% of row is black
                    bottom_boundary = h - y
                else:
                    if y > 10:  # Found content
                        bottom_boundary = h - y
                        break
            
            # LEFT edge
            left_boundary = 0
            for x in range(min(w//2, 300)):
                col = gray[:, x]
                black_pixels = np.sum(col < threshold)
                if black_pixels > h * 0.8:  # 80% of column is black
                    left_boundary = x
                else:
                    if x > 10:  # Found content
                        left_boundary = x
                        break
            
            # RIGHT edge
            right_boundary = w
            for x in range(min(w//2, 300)):
                col = gray[:, w-1-x]
                black_pixels = np.sum(col < threshold)
                if black_pixels > w * 0.8:  # 80% of column is black
                    right_boundary = w - x
                else:
                    if x > 10:  # Found content
                        right_boundary = w - x
                        break
            
            print(f"[{VERSION}] Edge scan results: T:{top_boundary}, B:{bottom_boundary}, L:{left_boundary}, R:{right_boundary}")
            
            # Method 2: Find the largest white/bright rectangle (content area)
            # This helps when edge scanning misses some black areas
            _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
            
            # Find contours of white areas
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find largest white area (should be our content)
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, cw, ch = cv2.boundingRect(largest_contour)
                
                # Update boundaries based on content area
                if x > left_boundary:
                    left_boundary = x
                if y > top_boundary:
                    top_boundary = y
                if x + cw < right_boundary:
                    right_boundary = x + cw
                if y + ch < bottom_boundary:
                    bottom_boundary = y + ch
                
                print(f"[{VERSION}] Content area found: ({x},{y}) size {cw}x{ch}")
            
            # Check if we found significant black frame
            frame_found = False
            if (top_boundary > 20 or (h - bottom_boundary) > 20 or 
                left_boundary > 20 or (w - right_boundary) > 20):
                frame_found = True
                
                # Add small margin inside the detected boundaries
                margin = 10
                crop_box = (
                    left_boundary + margin,
                    top_boundary + margin,
                    right_boundary - margin,
                    bottom_boundary - margin
                )
                
                # Ensure valid crop
                if crop_box[2] > crop_box[0] + 100 and crop_box[3] > crop_box[1] + 100:
                    print(f"[{VERSION}] Black frame removed: cropping to ({crop_box[0]},{crop_box[1]}) - ({crop_box[2]},{crop_box[3]})")
                    cropped = img_np[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
                    return Image.fromarray(cropped), True
            
            # Method 3: If still has issues, try morphological operations
            if not frame_found:
                # Use morphology to find the main content area
                kernel = np.ones((20, 20), np.uint8)
                closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                
                # Find contours again
                contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    x, y, cw, ch = cv2.boundingRect(largest)
                    
                    # Crop to this area
                    margin = 5
                    crop_box = (
                        max(0, x - margin),
                        max(0, y - margin),
                        min(w, x + cw + margin),
                        min(h, y + ch + margin)
                    )
                    
                    if crop_box[2] - crop_box[0] > 100 and crop_box[3] - crop_box[1] > 100:
                        print(f"[{VERSION}] Using morphological detection: cropping to content area")
                        cropped = img_np[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
                        return Image.fromarray(cropped), True
            
            print(f"[{VERSION}] No black rectangle frame detected")
            return image, False
            
        except Exception as e:
            print(f"[{VERSION}] Error in black frame detection: {e}")
            traceback.print_exc()
            return image, False
    
    def apply_enhancement_matching_v30(self, image):
        """Enhancement matching v30 enhancement handler"""
        try:
            # Match v30 enhancement settings - prioritize sharpness
            # 1. Pre-sharpening
            image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=100, threshold=1))
            
            # 2. Brightness
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.15)  # Match v30
            
            # 3. Contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.22)  # Match v30
            
            # 4. Saturation
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(0.94)  # Match v30
            
            # 5. Convert to numpy for advanced processing
            img_np = np.array(image)
            h, w = img_np.shape[:2]
            
            # 6. Apply CLAHE with v30 settings
            img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(img_lab)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l_channel = clahe.apply(l_channel)
            l_channel = cv2.add(l_channel, 8)
            l_channel = np.clip(l_channel, 0, 255)
            
            img_lab = cv2.merge([l_channel, a_channel, b_channel])
            img_np = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
            
            # 7. Clean white background
            background_color = (251, 250, 249)
            
            # Edge detection
            edges = cv2.Canny(cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY), 50, 150)
            edges_dilated = cv2.dilate(edges, np.ones((5,5), np.uint8), iterations=3)
            
            mask = np.ones((h, w), dtype=np.float32)
            mask[edges_dilated > 0] = 0
            mask = cv2.GaussianBlur(mask, (31, 31), 15)
            
            for i in range(3):
                img_np[:, :, i] = img_np[:, :, i] * (1 - mask * 0.12) + background_color[i] * mask * 0.12
            
            # 8. Gamma correction
            gamma = 0.92
            img_np = np.power(img_np / 255.0, gamma) * 255
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            
            # 9. Detail enhancement
            img_float = img_np.astype(np.float32)
            blurred = cv2.GaussianBlur(img_float, (0, 0), 3)
            detail = img_float - blurred
            img_float = img_float + detail * 0.5
            img_np = np.clip(img_float, 0, 255).astype(np.uint8)
            
            return Image.fromarray(img_np)
            
        except Exception as e:
            print(f"[{VERSION}] Error in enhancement: {e}")
            return image
    
    def create_super_tight_thumbnail_1000x1300(self, image):
        """Create super tight 1000x1300 thumbnail like image 5"""
        try:
            target_size = (1000, 1300)
            
            # Find wedding ring area with multiple methods
            img_np = np.array(image)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
            print(f"[{VERSION}] Finding rings for super tight crop...")
            
            # Method 1: Find metallic/bright areas (rings are usually bright)
            # Use adaptive threshold for better ring detection
            bright_thresh = cv2.adaptiveThreshold(gray, 255, 
                                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, 21, -2)
            
            # Method 2: Edge detection with lower threshold for ring details
            edges = cv2.Canny(gray, 30, 90)
            
            # Method 3: Find circular/elliptical shapes (rings)
            # Apply Hough Circle detection
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50,
                                     param1=50, param2=30, minRadius=20, maxRadius=200)
            
            # Combine all methods
            combined = cv2.bitwise_or(bright_thresh, edges)
            
            # If circles found, add them to combined mask
            if circles is not None:
                circles = np.uint16(np.around(circles))
                circle_mask = np.zeros_like(gray)
                for circle in circles[0, :]:
                    cv2.circle(circle_mask, (circle[0], circle[1]), circle[2], 255, -1)
                combined = cv2.bitwise_or(combined, circle_mask)
            
            # Clean up with morphology
            kernel = np.ones((3, 3), np.uint8)
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
            combined = cv2.dilate(combined, kernel, iterations=2)
            
            # Find all contours
            contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            ring_box = None
            if contours:
                # Filter contours by properties
                valid_contours = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 300:  # Minimum area
                        # Check shape characteristics
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                            
                            # Also check aspect ratio
                            x, y, w, h = cv2.boundingRect(contour)
                            aspect_ratio = w / h if h > 0 else 0
                            
                            # Rings are somewhat circular and have reasonable aspect ratio
                            if circularity > 0.2 or (0.5 < aspect_ratio < 2.0):
                                valid_contours.append(contour)
                
                if valid_contours:
                    # Get tight bounding box of all rings
                    all_points = np.concatenate(valid_contours)
                    x, y, w_box, h_box = cv2.boundingRect(all_points)
                    
                    # SUPER TIGHT padding - only 2% like image 5
                    padding_percent = 0.02  # Reduced from 8% to 2%
                    padding_x = int(w_box * padding_percent)
                    padding_y = int(h_box * padding_percent)
                    
                    x = max(0, x - padding_x)
                    y = max(0, y - padding_y)
                    w_box = min(img_np.shape[1] - x, w_box + 2 * padding_x)
                    h_box = min(img_np.shape[0] - y, h_box + 2 * padding_y)
                    
                    ring_box = (x, y, w_box, h_box)
                    print(f"[{VERSION}] Rings found at ({x},{y}) size {w_box}x{h_box}")
            
            # If no rings found by contours, try center crop
            if ring_box is None:
                print(f"[{VERSION}] Using center crop fallback")
                h, w = image.size[1], image.size[0]
                
                # Assume rings are in center 40% (tighter than before)
                margin_x = int(w * 0.3)  # 30% margin each side
                margin_y = int(h * 0.3)
                ring_box = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)
            
            # Crop to ring area
            x, y, w_box, h_box = ring_box
            cropped = image.crop((x, y, x + w_box, y + h_box))
            
            print(f"[{VERSION}] Cropped to ring area: {cropped.size}")
            
            # Calculate scale to FILL 1000x1300 completely
            cropped_w, cropped_h = cropped.size
            
            # We want to fill the entire frame, so use max scale
            scale_w = target_size[0] / cropped_w
            scale_h = target_size[1] / cropped_h
            scale = max(scale_w, scale_h)  # This ensures we fill the frame
            
            # Apply scale with some extra to ensure no borders
            scale *= 1.05  # 5% extra to ensure full coverage
            
            new_w = int(cropped_w * scale)
            new_h = int(cropped_h * scale)
            
            print(f"[{VERSION}] Scaling from {cropped.size} to ({new_w},{new_h})")
            
            # High quality resize
            resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Center crop to exact 1000x1300
            if new_w > target_size[0] or new_h > target_size[1]:
                # Center crop
                left = (new_w - target_size[0]) // 2
                top = (new_h - target_size[1]) // 2
                final = resized.crop((left, top, left + target_size[0], top + target_size[1]))
            else:
                # This shouldn't happen with our scale calculation, but just in case
                final = Image.new('RGB', target_size, (251, 250, 249))
                paste_x = (target_size[0] - new_w) // 2
                paste_y = (target_size[1] - new_h) // 2
                final.paste(resized, (paste_x, paste_y))
            
            # Strong detail enhancement for crisp look
            # Multiple sharpening passes
            enhancer = ImageEnhance.Sharpness(final)
            final = enhancer.enhance(1.5)
            
            # Additional detail enhancement
            final_np = np.array(final)
            
            # Unsharp mask
            gaussian = cv2.GaussianBlur(final_np, (0, 0), 2.0)
            unsharp = cv2.addWeighted(final_np, 1.6, gaussian, -0.6, 0)
            
            # Local contrast enhancement
            kernel = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]]) / 8.0
            
            enhanced = cv2.filter2D(unsharp, -1, kernel)
            final_np = cv2.addWeighted(unsharp, 0.7, enhanced, 0.3, 0)
            
            # Ensure no over-sharpening
            final_np = np.clip(final_np, 0, 255).astype(np.uint8)
            
            print(f"[{VERSION}] Created super tight 1000x1300 thumbnail")
            return Image.fromarray(final_np)
            
        except Exception as e:
            print(f"[{VERSION}] Error creating thumbnail: {e}")
            # Fallback to simple resize
            return image.resize((1000, 1300), Image.Resampling.LANCZOS)

def handler(job):
    """RunPod handler - V30 with perfect detection and super tight crop"""
    print(f"[{VERSION}] ====== Thumbnail Handler Started ======")
    print(f"[{VERSION}] Perfect Detection & Super Tight Crop")
    
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
        processor = ThumbnailProcessorV30()
        
        # Process image step by step
        had_black_frame = False
        
        # 1. COMPLETE black rectangle detection (including bottom)
        try:
            image, had_black_frame = processor.detect_black_rectangle_complete(image)
            print(f"[{VERSION}] Black rectangle detection complete: {had_black_frame}")
        except Exception as e:
            print(f"[{VERSION}] Error in black frame detection: {e}")
        
        # 2. Apply color enhancement matching v30
        try:
            image = processor.apply_enhancement_matching_v30(image)
            print(f"[{VERSION}] Color enhancement applied (v30 style)")
        except Exception as e:
            print(f"[{VERSION}] Error in color enhancement: {e}")
        
        # 3. Create SUPER TIGHT 1000x1300 thumbnail
        try:
            thumbnail = processor.create_super_tight_thumbnail_1000x1300(image)
            print(f"[{VERSION}] Super tight thumbnail created: {thumbnail.size}")
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
                "processing_method": "complete_detection_super_tight_v30",
                "processing_time": round(processing_time, 2),
                "warning": "Google Script must add padding: while (base64Data.length % 4 !== 0) { base64Data += '='; }"
            }
        }
        
        print(f"[{VERSION}] ====== Success - Returning Super Tight Thumbnail ======")
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
    print("V30 - Complete Detection & Super Tight Crop")
    print("Features:")
    print("- Complete edge scanning (especially bottom)")
    print("- Content area detection")
    print("- Morphological operations fallback")
    print("- Super tight crop (2% padding)")
    print("- Multiple ring detection methods")
    print("- Strong detail enhancement")
    print("CRITICAL: Padding is removed for Make.com")
    print("Google Apps Script MUST add padding back:")
    print("while (base64Data.length % 4 !== 0) { base64Data += '='; }")
    print("="*70)
    
    runpod.serverless.start({"handler": handler})
