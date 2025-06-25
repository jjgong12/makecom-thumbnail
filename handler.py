import runpod
import base64
from io import BytesIO
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import traceback

VERSION = "thumbnail_v51"
print(f"{VERSION} starting...")

def find_input_url(event):
    """Find URL or base64 data from various possible locations"""
    print(f"Searching for image in event with keys: {list(event.keys())}")
    
    # PRIORITY 1: Check direct enhanced_image paths first
    priority_paths = [
        'enhanced_image',
        'input.enhanced_image',
        'input.4.data.output.output.enhanced_image'
    ]
    
    for path in priority_paths:
        keys = path.split('.')
        value = event
        try:
            for key in keys:
                if isinstance(value, dict):
                    value = value.get(key)
                else:
                    value = None
                    break
            if value and isinstance(value, str) and len(value) > 100:
                print(f"Found enhanced_image at: {path}")
                return value
        except:
            continue
    
    # PRIORITY 2: Check numbered patterns (Make.com structure)
    # First check if input exists and has numbered keys
    if 'input' in event and isinstance(event['input'], dict):
        for i in range(10):
            # Check input.{i}.data.output.output.enhanced_image
            if str(i) in event['input']:
                try:
                    node = event['input'][str(i)]
                    if isinstance(node, dict) and 'data' in node:
                        data = node['data']
                        if isinstance(data, dict) and 'output' in data:
                            output = data['output']
                            if isinstance(output, dict) and 'output' in output:
                                inner_output = output['output']
                                if isinstance(inner_output, dict) and 'enhanced_image' in inner_output:
                                    value = inner_output['enhanced_image']
                                    if value and isinstance(value, str):
                                        print(f"Found enhanced_image at: input.{i}.data.output.output.enhanced_image")
                                        return value
                except:
                    continue
    
    # Check root level numbered patterns
    for i in range(10):
        if str(i) in event:
            try:
                node = event[str(i)]
                if isinstance(node, dict) and 'data' in node:
                    data = node['data']
                    if isinstance(data, dict) and 'output' in data:
                        output = data['output']
                        if isinstance(output, dict) and 'output' in output:
                            inner_output = output['output']
                            if isinstance(inner_output, dict) and 'enhanced_image' in inner_output:
                                value = inner_output['enhanced_image']
                                if value and isinstance(value, str):
                                    print(f"Found enhanced_image at: {i}.data.output.output.enhanced_image")
                                    return value
            except:
                continue
    
    # PRIORITY 3: Check other image paths
    fallback_paths = [
        'image_url', 'url', 'imageUrl', 'image', 'image_base64',
        'input.image_url', 'input.url', 'input.image', 'input.image_base64'
    ]
    
    for path in fallback_paths:
        keys = path.split('.')
        value = event
        try:
            for key in keys:
                if isinstance(value, dict):
                    value = value.get(key)
                else:
                    value = None
                    break
            if value and isinstance(value, str) and (
                value.startswith('http') or 
                value.startswith('data:') or
                len(value) > 100
            ):
                print(f"Found image data at: {path}")
                return value
        except:
            continue
    
    # PRIORITY 4: Deep search in event structure
    def search_dict(d, depth=0, max_depth=5):
        if depth > max_depth:
            return None
        if not isinstance(d, dict):
            return None
            
        # Check for enhanced_image key
        if 'enhanced_image' in d:
            value = d['enhanced_image']
            if isinstance(value, str) and len(value) > 100:
                return value
        
        # Check for any image-related keys
        for key in ['image', 'image_url', 'url', 'image_base64']:
            if key in d:
                value = d[key]
                if isinstance(value, str) and len(value) > 100:
                    return value
        
        # Recursive search
        for key, value in d.items():
            if isinstance(value, dict):
                result = search_dict(value, depth + 1, max_depth)
                if result:
                    return result
        
        return None
    
    # Try deep search
    result = search_dict(event)
    if result:
        print("Found image data through deep search")
        return result
    
    print(f"No image data found. Full event structure: {event}")
    return None

def detect_wedding_rings(img_np):
    """Detect wedding rings using multiple methods"""
    h, w = img_np.shape[:2]
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Method 1: Find bright metallic regions
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    
    # Method 2: Find circular shapes
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=20,
        maxRadius=min(w, h)//3
    )
    
    # Method 3: Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Default to center
    best_x = w // 2
    best_y = h // 2
    best_size = min(w, h) // 3
    
    # Use circles if found
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            cx, cy, r = circle
            if abs(cx - w//2) < w//3 and abs(cy - h//2) < h//3:
                best_x = cx
                best_y = cy
                best_size = r * 2
                break
    
    # Check contours if no circles
    elif len(contours) > 0:
        max_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area and area > 100:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    if abs(cx - w//2) < w//3 and abs(cy - h//2) < h//3:
                        x, y, w_c, h_c = cv2.boundingRect(contour)
                        best_x = cx
                        best_y = cy
                        best_size = max(w_c, h_c)
                        max_area = area
    
    # Return bounding box
    x1 = max(0, best_x - best_size)
    y1 = max(0, best_y - best_size)
    x2 = min(w, best_x + best_size)
    y2 = min(h, best_y + best_size)
    
    return x1, y1, x2, y2

def detect_metal_type(img_np):
    """Detect metal type based on color"""
    avg_color = img_np.mean(axis=(0, 1))
    r, g, b = avg_color
    
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    avg_hsv = hsv.mean(axis=(0, 1))
    h, s, v = avg_hsv
    
    # Detect metal type
    if v > 200 and s < 30:
        return "white"
    elif h < 20 and s > 40 and r > g:
        return "rose"
    elif 20 < h < 40 and s > 30:
        return "yellow"
    else:
        return "white"

def apply_metal_enhancement(img, metal_type):
    """Apply metal-specific enhancements"""
    img_np = np.array(img)
    
    if metal_type == "white":
        # Bright and cool
        img_np = cv2.convertScaleAbs(img_np, alpha=1.25, beta=25)
        img_np[:,:,0] = np.clip(img_np[:,:,0] * 0.97, 0, 255)
        img_np[:,:,2] = np.clip(img_np[:,:,2] * 1.03, 0, 255)
        
    elif metal_type == "rose":
        # Warm rose tones
        img_np = cv2.convertScaleAbs(img_np, alpha=1.15, beta=15)
        img_np[:,:,0] = np.clip(img_np[:,:,0] * 1.08, 0, 255)
        img_np[:,:,1] = np.clip(img_np[:,:,1] * 1.03, 0, 255)
        
    elif metal_type == "yellow":
        # Golden tones
        img_np = cv2.convertScaleAbs(img_np, alpha=1.18, beta=18)
        img_np[:,:,0] = np.clip(img_np[:,:,0] * 1.05, 0, 255)
        img_np[:,:,1] = np.clip(img_np[:,:,1] * 1.08, 0, 255)
    
    return Image.fromarray(img_np.astype(np.uint8))

def enhance_details(img):
    """Final detail enhancement"""
    img_np = np.array(img)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoisingColored(img_np, None, 3, 3, 7, 21)
    
    # Sharpen
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) / 9.0
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    # Blend
    result = cv2.addWeighted(denoised, 0.6, sharpened, 0.4, 0)
    
    # Convert back
    img = Image.fromarray(result)
    
    # Final adjustments
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.3)
    
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.15)
    
    return img

def thumbnail_handler(event):
    print(f"=== {VERSION} Handler Started ===")
    print(f"Event keys: {list(event.keys())}")
    if 'input' in event and isinstance(event['input'], dict):
        print(f"Input keys: {list(event['input'].keys())}")
    
    try:
        # Find image data
        img_data = find_input_url(event)
        if not img_data:
            return {
                "output": {
                    "error": "No image URL or data found",
                    "status": "error",
                    "version": VERSION,
                    "debug_info": {
                        "event_keys": list(event.keys()),
                        "input_keys": list(event.get('input', {}).keys()) if 'input' in event else None
                    }
                }
            }
        
        print(f"Found image data, type: {'data URL' if img_data.startswith('data:') else 'base64'}")
        print(f"Data length: {len(img_data)}")
        
        # Extract base64 data
        if img_data.startswith('data:'):
            base64_data = img_data.split(',')[1]
        else:
            base64_data = img_data
        
        # IMPORTANT: Add padding for decoding
        padding = 4 - (len(base64_data) % 4)
        if padding != 4:
            base64_data += '=' * padding
        
        # Decode
        try:
            img_bytes = base64.b64decode(base64_data)
        except Exception as e:
            print(f"Base64 decode error: {str(e)}")
            return {
                "output": {
                    "error": f"Base64 decode error: {str(e)}",
                    "status": "error",
                    "version": VERSION
                }
            }
        
        # Open image
        img = Image.open(BytesIO(img_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        original_img = img.copy()
        orig_w, orig_h = img.size
        print(f"Image opened: {orig_w}x{orig_h}")
        
        # Step 1: Create detection image
        max_detect_size = 1500
        if orig_w > max_detect_size or orig_h > max_detect_size:
            ratio = min(max_detect_size/orig_w, max_detect_size/orig_h)
            detect_w = int(orig_w * ratio)
            detect_h = int(orig_h * ratio)
            detect_img = img.resize((detect_w, detect_h), Image.Resampling.LANCZOS)
        else:
            detect_img = img
            ratio = 1.0
        
        # Step 2: Detect rings
        detect_np = np.array(detect_img)
        x1, y1, x2, y2 = detect_wedding_rings(detect_np)
        
        # Step 3: Scale coordinates
        orig_x1 = int(x1 / ratio)
        orig_y1 = int(y1 / ratio)
        orig_x2 = int(x2 / ratio)
        orig_y2 = int(y2 / ratio)
        
        # Step 4: Add padding
        pad_x = int((orig_x2 - orig_x1) * 0.3)
        pad_y = int((orig_y2 - orig_y1) * 0.3)
        
        crop_x1 = max(0, orig_x1 - pad_x)
        crop_y1 = max(0, orig_y1 - pad_y)
        crop_x2 = min(orig_w, orig_x2 + pad_x)
        crop_y2 = min(orig_h, orig_y2 + pad_y)
        
        # Step 5: Crop
        cropped = original_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        
        # Step 6: Detect metal
        crop_np = np.array(cropped)
        metal_type = detect_metal_type(crop_np)
        print(f"Detected metal: {metal_type}")
        
        # Step 7: Resize
        thumb = cropped.resize((1000, 1300), Image.Resampling.LANCZOS)
        
        # Step 8: Apply enhancements
        thumb = apply_metal_enhancement(thumb, metal_type)
        thumb = enhance_details(thumb)
        
        # Save as JPEG
        output_buffer = BytesIO()
        thumb.save(output_buffer, format='JPEG', quality=95)
        thumb_bytes = output_buffer.getvalue()
        
        # IMPORTANT: Encode and remove padding for Make.com
        thumb_base64 = base64.b64encode(thumb_bytes).decode('utf-8').rstrip('=')
        
        print(f"{VERSION} completed successfully")
        
        return {
            "output": {
                "thumbnail": f"data:image/jpeg;base64,{thumb_base64}",
                "status": "success",
                "version": VERSION,
                "metal_type": metal_type,
                "ring_detected": True,
                "crop_area": f"{crop_x1},{crop_y1},{crop_x2},{crop_y2}",
                "final_size": "1000x1300"
            }
        }
        
    except Exception as e:
        print(f"Error in {VERSION}: {str(e)}")
        print(traceback.format_exc())
        return {
            "output": {
                "error": str(e),
                "status": "error",
                "version": VERSION,
                "traceback": traceback.format_exc()
            }
        }

runpod.serverless.start({"handler": thumbnail_handler})
