import runpod
import base64
from io import BytesIO
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import traceback

VERSION = "thumbnail_v48"
print(f"{VERSION} starting...")

def find_input_url(event):
    """Find URL from various possible locations"""
    # Direct URL paths
    url_paths = [
        'enhanced_image', 'image_url', 'url', 'imageUrl',
        'input.enhanced_image', 'input.image_url', 'input.url'
    ]
    
    for path in url_paths:
        try:
            keys = path.split('.')
            value = event
            for key in keys:
                value = value.get(key, {})
            if value and isinstance(value, str) and (
                value.startswith('http') or 
                value.startswith('data:') or
                (len(value) > 100 and not ' ' in value)
            ):
                return value
        except:
            continue
    
    # Check numbered patterns
    for i in range(10):
        paths = [
            f"{i}.data.output.output.enhanced_image",
            f"input.{i}.data.output.output.enhanced_image"
        ]
        for path in paths:
            try:
                keys = path.split('.')
                value = event
                for key in keys:
                    value = value.get(str(key), {})
                if value and isinstance(value, str):
                    return value
            except:
                continue
    
    return None

def detect_wedding_rings(img_np):
    """Detect wedding rings using multiple methods on resized image"""
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
    
    # Method 3: Find contours of metallic objects
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Default to center region if no specific detection
    best_x = w // 2
    best_y = h // 2
    best_size = min(w, h) // 3
    
    # If circles detected, use the most prominent one
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            cx, cy, r = circle
            # Prefer circles near center
            if abs(cx - w//2) < w//3 and abs(cy - h//2) < h//3:
                best_x = cx
                best_y = cy
                best_size = r * 2
                break
    
    # If no circles, check contours
    elif len(contours) > 0:
        # Find largest contour near center
        best_contour = None
        max_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    # Check if near center
                    if abs(cx - w//2) < w//3 and abs(cy - h//2) < h//3:
                        max_area = area
                        best_contour = contour
                        best_x = cx
                        best_y = cy
        
        if best_contour is not None:
            x, y, w_c, h_c = cv2.boundingRect(best_contour)
            best_size = max(w_c, h_c)
    
    # Return bounding box
    x1 = max(0, best_x - best_size)
    y1 = max(0, best_y - best_size)
    x2 = min(w, best_x + best_size)
    y2 = min(h, best_y + best_size)
    
    return x1, y1, x2, y2

def detect_metal_type(img_np):
    """Detect the type of metal based on color analysis"""
    # Calculate average color
    avg_color = img_np.mean(axis=(0, 1))
    r, g, b = avg_color
    
    # Convert to HSV for better analysis
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    avg_hsv = hsv.mean(axis=(0, 1))
    h, s, v = avg_hsv
    
    # Determine metal type
    if v > 200 and s < 30:
        return "white"  # White gold or unplated white
    elif h < 20 and s > 40 and r > g:
        return "rose"  # Rose gold
    elif 20 < h < 40 and s > 30:
        return "yellow"  # Yellow gold
    else:
        return "white"  # Default to white

def apply_metal_specific_enhancement(img, metal_type):
    """Apply enhancement based on detected metal type"""
    img_np = np.array(img)
    
    if metal_type == "white":
        # Make whites brighter and cooler
        img_np = cv2.convertScaleAbs(img_np, alpha=1.25, beta=25)
        # Cool tone adjustment
        img_np[:,:,0] = np.clip(img_np[:,:,0] * 0.97, 0, 255)  # Reduce red
        img_np[:,:,2] = np.clip(img_np[:,:,2] * 1.03, 0, 255)  # Increase blue
        
    elif metal_type == "rose":
        # Enhance rose gold warmth
        img_np = cv2.convertScaleAbs(img_np, alpha=1.15, beta=15)
        # Warm tone adjustment
        img_np[:,:,0] = np.clip(img_np[:,:,0] * 1.08, 0, 255)  # Increase red
        img_np[:,:,1] = np.clip(img_np[:,:,1] * 1.03, 0, 255)  # Slight green
        
    elif metal_type == "yellow":
        # Enhance yellow gold
        img_np = cv2.convertScaleAbs(img_np, alpha=1.18, beta=18)
        # Golden tone adjustment
        img_np[:,:,0] = np.clip(img_np[:,:,0] * 1.05, 0, 255)  # Increase red
        img_np[:,:,1] = np.clip(img_np[:,:,1] * 1.08, 0, 255)  # Increase green
    
    return Image.fromarray(img_np.astype(np.uint8))

def enhance_details(img):
    """Enhance details after resizing"""
    # Convert to numpy
    img_np = np.array(img)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoisingColored(img_np, None, 3, 3, 7, 21)
    
    # Sharpen
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]]) / 9.0
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    # Blend
    result = cv2.addWeighted(denoised, 0.6, sharpened, 0.4, 0)
    
    # Convert back to PIL
    img = Image.fromarray(result)
    
    # Final adjustments
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.3)
    
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.15)
    
    return img

def thumbnail_handler(event):
    print(f"=== {VERSION} Handler Started ===")
    
    try:
        # Find image URL or base64
        img_data = find_input_url(event)
        if not img_data:
            return {
                "output": {
                    "error": "No image URL found",
                    "status": "error",
                    "version": VERSION
                }
            }
        
        # Handle base64 data
        if img_data.startswith('data:'):
            base64_data = img_data.split(',')[1]
        else:
            base64_data = img_data
        
        # Add padding for decoding
        padding = 4 - (len(base64_data) % 4)
        if padding != 4:
            base64_data += '=' * padding
        
        img_bytes = base64.b64decode(base64_data)
        
        # Open image
        img = Image.open(BytesIO(img_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        original_img = img.copy()
        orig_w, orig_h = img.size
        
        print(f"Original image size: {orig_w}x{orig_h}")
        
        # Step 1: Create detection image (max 1500px for speed)
        max_detect_size = 1500
        if orig_w > max_detect_size or orig_h > max_detect_size:
            ratio = min(max_detect_size/orig_w, max_detect_size/orig_h)
            detect_w = int(orig_w * ratio)
            detect_h = int(orig_h * ratio)
            detect_img = img.resize((detect_w, detect_h), Image.Resampling.LANCZOS)
        else:
            detect_img = img
            ratio = 1.0
        
        # Step 2: Detect rings on smaller image
        detect_np = np.array(detect_img)
        x1, y1, x2, y2 = detect_wedding_rings(detect_np)
        
        # Step 3: Convert coordinates to original size
        orig_x1 = int(x1 / ratio)
        orig_y1 = int(y1 / ratio)
        orig_x2 = int(x2 / ratio)
        orig_y2 = int(y2 / ratio)
        
        # Step 4: Add padding for better composition
        pad_x = int((orig_x2 - orig_x1) * 0.3)
        pad_y = int((orig_y2 - orig_y1) * 0.3)
        
        crop_x1 = max(0, orig_x1 - pad_x)
        crop_y1 = max(0, orig_y1 - pad_y)
        crop_x2 = min(orig_w, orig_x2 + pad_x)
        crop_y2 = min(orig_h, orig_y2 + pad_y)
        
        # Step 5: Crop from original
        cropped = original_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        
        # Step 6: Detect metal type
        crop_np = np.array(cropped)
        metal_type = detect_metal_type(crop_np)
        print(f"Detected metal type: {metal_type}")
        
        # Step 7: Resize to 1000x1300
        thumb = cropped.resize((1000, 1300), Image.Resampling.LANCZOS)
        
        # Step 8: Apply metal-specific enhancement
        thumb = apply_metal_specific_enhancement(thumb, metal_type)
        
        # Step 9: Enhance details
        thumb = enhance_details(thumb)
        
        # Save as JPEG
        output_buffer = BytesIO()
        thumb.save(output_buffer, format='JPEG', quality=95)
        thumb_bytes = output_buffer.getvalue()
        
        # Encode to base64 and remove padding for Make.com
        thumb_base64 = base64.b64encode(thumb_bytes).decode('utf-8')
        thumb_base64 = thumb_base64.rstrip('=')
        
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
                "version": VERSION
            }
        }

runpod.serverless.start({"handler": thumbnail_handler})
