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
VERSION = "v27-thumbnail"

class ThumbnailProcessorV27:
    """v27 Thumbnail Processor - Rectangle Frame Detection"""
    
    def __init__(self):
        print(f"[{VERSION}] Initializing - Rectangle Frame Detection")
    
    def detect_black_rectangle_frame(self, image):
        """Detect black rectangle frame (ㅁ shape) accurately"""
        img_np = np.array(image)
        h, w = img_np.shape[:2]
        print(f"[{VERSION}] Rectangle frame detection - Processing {w}x{h} image")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Method 1: Find rectangle contours
        # Use multiple thresholds to find the best rectangle
        best_rect = None
        best_score = 0
        
        for thresh_val in [30, 40, 50, 60, 70]:
            # Create binary image
            _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Get bounding rectangle
                x, y, cw, ch = cv2.boundingRect(contour)
                
                # Check if it's a significant rectangle
                area = cw * ch
                if area < (w * h * 0.1):  # Too small
                    continue
                
                # Check if contour is rectangular
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # A rectangle should have 4 vertices
                if len(approx) == 4:
                    # Calculate how "rectangular" this shape is
                    rect_area = cw * ch
                    contour_area = cv2.contourArea(contour)
                    rectangularity = contour_area / rect_area if rect_area > 0 else 0
                    
                    # Check if it forms a frame (hollow rectangle)
                    # by checking if center is brighter than edges
                    center_x, center_y = x + cw//2, y + ch//2
                    center_brightness = np.mean(gray[center_y-10:center_y+10, center_x-10:center_x+10])
                    edge_brightness = np.mean(gray[y:y+20, x:x+cw])
                    
                    if center_brightness > edge_brightness + 20:  # Center is brighter
                        score = rectangularity * (area / (w * h))
                        if score > best_score:
                            best_score = score
                            best_rect = (x, y, cw, ch)
                            print(f"[{VERSION}] Found rectangle frame at ({x},{y}) size {cw}x{ch}, score: {score:.3f}")
        
        # Method 2: Detect rectangle by finding lines
        if best_rect is None:
            print(f"[{VERSION}] Trying line detection method")
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find lines using HoughLinesP
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=20)
            
            if lines is not None:
                # Separate horizontal and vertical lines
                h_lines = []
                v_lines = []
                
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    
                    # Check if horizontal
                    if abs(y2 - y1) < 10 and abs(x2 - x1) > 50:
                        h_lines.append((y1 + y2) // 2)
                    # Check if vertical
                    elif abs(x2 - x1) < 10 and abs(y2 - y1) > 50:
                        v_lines.append((x1 + x2) // 2)
                
                # Find rectangle formed by lines
                if len(h_lines) >= 2 and len(v_lines) >= 2:
                    h_lines.sort()
                    v_lines.sort()
                    
                    # Find the outermost rectangle
                    top = min(h_lines)
                    bottom = max(h_lines)
                    left = min(v_lines)
                    right = max(v_lines)
                    
                    # Verify it's a valid rectangle
                    if bottom - top > 100 and right - left > 100:
                        best_rect = (left, top, right - left, bottom - top)
                        print(f"[{VERSION}] Found rectangle from lines: {best_rect}")
        
        # Method 3: Scan for black frame from edges
        if best_rect is None:
            print(f"[{VERSION}] Using edge scanning method")
            
            # Find where the black frame ends
            threshold = 60
            
            # Scan from edges
            top, bottom, left, right = 0, h, 0, w
            
            # Top edge
            for y in range(h//3):
                if np.mean(gray[y, w//4:3*w//4]) < threshold:
                    top = y
                else:
                    if y > 10:  # Found transition
                        top = y
                        break
            
            # Bottom edge
            for y in range(h//3):
                if np.mean(gray[h-1-y, w//4:3*w//4]) < threshold:
                    bottom = h - y
                else:
                    if y > 10:
                        bottom = h - y
                        break
            
            # Left edge
            for x in range(w//3):
                if np.mean(gray[h//4:3*h//4, x]) < threshold:
                    left = x
                else:
                    if x > 10:
                        left = x
                        break
            
            # Right edge
            for x in range(w//3):
                if np.mean(gray[h//4:3*h//4, w-1-x]) < threshold:
                    right = w - x
                else:
                    if x > 10:
                        right = w - x
                        break
            
            if top > 0 or bottom < h or left > 0 or right < w:
                best_rect = (left, top, right - left, bottom - top)
                print(f"[{VERSION}] Found frame by scanning: {best_rect}")
        
        # Apply the best rectangle found
        if best_rect:
            x, y, cw, ch = best_rect
            
            # Add small margin inside the black frame
            margin = 10
            x += margin
            y += margin
            cw -= 2 * margin
            ch -= 2 * margin
            
            # Ensure valid dimensions
            x = max(0, x)
            y = max(0, y)
            cw = min(w - x, cw)
            ch = min(h - y, ch)
            
            print(f"[{VERSION}] Final crop rectangle: ({x},{y}) size {cw}x{ch}")
            
            # Crop the image
            cropped = img_np[y:y+ch, x:x+cw]
            return Image.fromarray(cropped), True
        
        print(f"[{VERSION}] No black rectangle frame detected")
        return image, False
    
    def apply_simple_enhancement(self, image):
        """Enhancement matching v27 enhancement handler"""
        # 1. Brightness
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.18)  # Match v27
        
        # 2. Contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.10)  # Match v27
        
        # 3. Saturation
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(0.95)  # Match v27
        
        # 4. Soft white background
        img_np = np.array(image)
        h, w = img_np.shape[:2]
        
        background_color = (248, 246, 243)
        
        # Gradient overlay
        y_gradient = np.linspace(0.7, 1.0, h).reshape((h, 1))
        x_gradient = np.linspace(1.0, 1.0, w).reshape((1, w))
        gradient_mask = y_gradient * x_gradient
        
        for i in range(3):
            overlay = background_color[i] * gradient_mask
            img_np[:, :, i] = img_np[:, :, i] * 0.88 + overlay * 0.12
        
        # Soft adjustments
        img_np = np.clip(img_np * 1.02, 0, 255)
        
        # Gamma correction
        gamma = 0.95
        img_np = np.power(img_np / 255.0, gamma) * 255
        img_np = np.clip(img_np, 0, 255)
        
        return Image.fromarray(img_np.astype(np.uint8))
    
    def create_thumbnail_1000x1300(self, image):
        """Create exact 1000x1300 thumbnail - focus on rings"""
        target_size = (1000, 1300)
        
        # Find wedding ring area
        img_np = np.array(image)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Find bright areas (rings)
        _, bright = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        
        # Edge detection with lower threshold for better ring detection
        edges = cv2.Canny(gray, 30, 80)
        
        # Combine methods
        combined = cv2.bitwise_or(bright, edges)
        
        # Clean up
        kernel = np.ones((5, 5), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Filter and find ring contours
            min_area = 200
            valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
            
            if valid_contours:
                # Get bounding box of all rings
                all_points = np.concatenate(valid_contours)
                x, y, w_box, h_box = cv2.boundingRect(all_points)
                
                # Tight padding (3%)
                padding_x = int(w_box * 0.03)
                padding_y = int(h_box * 0.03)
                
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
        else:
            # Fallback: aggressive center crop
            w, h = image.size
            margin_x = int(w * 0.25)
            margin_y = int(h * 0.25)
            cropped = image.crop((margin_x, margin_y, w - margin_x, h - margin_y))
        
        # Resize to fill 1000x1300
        cropped_w, cropped_h = cropped.size
        scale = max(1000 / cropped_w, 1300 / cropped_h)
        
        new_w = int(cropped_w * scale)
        new_h = int(cropped_h * scale)
        resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Create canvas
        canvas = Image.new('RGB', target_size, (248, 246, 243))
        
        # Center crop or paste
        if new_w > 1000 or new_h > 1300:
            left = (new_w - 1000) // 2
            top = (new_h - 1300) // 2
            resized = resized.crop((left, top, left + 1000, top + 1300))
            canvas = resized
        else:
            paste_x = (1000 - new_w) // 2
            paste_y = (1300 - new_h) // 2
            canvas.paste(resized, (paste_x, paste_y))
        
        # Moderate sharpness
        enhancer = ImageEnhance.Sharpness(canvas)
        canvas = enhancer.enhance(1.6)  # Reduced from 2.0
        
        print(f"[{VERSION}] Created exact 1000x1300 thumbnail")
        return canvas

def handler(job):
    """RunPod handler - V27 with rectangle frame detection"""
    print(f"[{VERSION}] ====== Thumbnail Handler Started ======")
    print(f"[{VERSION}] Rectangle Frame Detection Active")
    print(f"[{VERSION}] CRITICAL: Google Apps Script requires padding!")
    
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
        processor = ThumbnailProcessorV27()
        
        # RECTANGLE frame detection
        image, had_black_frame = processor.detect_black_rectangle_frame(image)
        
        if had_black_frame:
            print(f"[{VERSION}] Black rectangle frame removed successfully")
        else:
            print(f"[{VERSION}] No black rectangle frame found")
        
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
                "has_black_frame": had_black_frame,
                "success": True,
                "version": VERSION,
                "thumbnail_size": [1000, 1300],
                "processing_method": "rectangle_frame_detection_v27",
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
    print("V27 - Rectangle Frame Detection (ㅁ shape)")
    print("CRITICAL: Padding is removed for Make.com")
    print("Google Apps Script MUST add padding back:")
    print("while (base64Data.length % 4 !== 0) { base64Data += '='; }")
    print("="*70)
    
    runpod.serverless.start({"handler": handler})
