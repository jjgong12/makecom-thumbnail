import cv2
import numpy as np
from PIL import Image
import base64
import io
import os
import replicate
import requests


def detect_metal_type(img):
    """Detect metal type for thumbnail processing"""
    try:
        h, w = img.shape[:2]
        center_y, center_x = h // 2, w // 2
        
        # Sample center region
        sample_size = min(h, w) // 4
        center_region = img[center_y-sample_size:center_y+sample_size,
                          center_x-sample_size:center_x+sample_size]
        
        # Convert to HSV
        hsv = cv2.cvtColor(center_region, cv2.COLOR_BGR2HSV)
        
        # Calculate averages
        avg_sat = np.mean(hsv[:, :, 1])
        avg_val = np.mean(hsv[:, :, 2])
        
        # Calculate RGB averages
        avg_b = np.mean(center_region[:, :, 0])
        avg_g = np.mean(center_region[:, :, 1])
        avg_r = np.mean(center_region[:, :, 2])
        
        # Metal detection
        if avg_sat < 30 and avg_val > 180:
            if avg_r > avg_b + 5:
                return 'champagne_gold'
            else:
                return 'white_gold'
        elif avg_r > avg_b + 10 and avg_sat > 30:
            return 'rose_gold'
        elif avg_r > avg_g + 5 and avg_g > avg_b + 5:
            return 'yellow_gold'
        else:
            return 'white_gold'
    except:
        return 'white_gold'

def enhance_ring_details(img, metal_type):
    """Enhance ring details specifically for thumbnail"""
    try:
        result = img.copy()
        
        # Strong sharpening for details
        kernel = np.array([[-1, -1, -1], 
                          [-1, 9.5, -1], 
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(result, -1, kernel)
        result = cv2.addWeighted(result, 0.3, sharpened, 0.7, 0)
        
        # Increase contrast significantly
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE with stronger parameters
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        l = clahe.apply(l)
        
        result = cv2.merge([l, a, b])
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        
        # Metal-specific enhancements
        if metal_type in ['white_gold', 'champagne_gold']:
            # Make it brighter and more white
            result = cv2.addWeighted(result, 0.85, np.ones_like(result) * 255, 0.15, 0)
            result = cv2.convertScaleAbs(result, alpha=1.2, beta=10)
        elif metal_type == 'rose_gold':
            # Enhance pink tones
            result[:, :, 2] = cv2.add(result[:, :, 2], 10)  # Add red
            result = cv2.convertScaleAbs(result, alpha=1.15, beta=5)
        else:  # yellow_gold
            # Enhance yellow tones
            result[:, :, 1] = cv2.add(result[:, :, 1], 8)   # Add green
            result[:, :, 2] = cv2.add(result[:, :, 2], 12)  # Add red
            result = cv2.convertScaleAbs(result, alpha=1.1, beta=5)
        
        return result
    except:
        return img

def find_ring_precise(img):
    """Find ring location with high precision"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Use multiple thresholds to find the ring
        thresholds = [240, 230, 220, 210, 200]
        best_contour = None
        max_area = 0
        
        for thresh in thresholds:
            _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
            
            # Morphological operations to clean up
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > max_area and area > 1000:  # Minimum area threshold
                    max_area = area
                    best_contour = contour
        
        if best_contour is not None:
            return cv2.boundingRect(best_contour)
        
        # Fallback: use edge detection
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            return cv2.boundingRect(largest)
        
        # Last resort: center crop
        h, w = img.shape[:2]
        return w//4, h//4, w//2, h//2
        
    except:
        h, w = img.shape[:2]
        return w//4, h//4, w//2, h//2

def remove_padding_safe(base64_string):
    """Remove base64 padding safely for Make.com compatibility"""
    return base64_string.rstrip('=')

def handler(event):
    """Thumbnail handler for wedding ring images"""
    try:
        # Extract image data
        input_data = event.get('input', {})
        image_data = input_data.get('image')
        
        if not image_data:
            return {"output": {"error": "No image provided in input", "status": "error"}}
        
        # Decode base64 image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to OpenCV format
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect metal type
        metal_type = detect_metal_type(img)
        print(f"Detected metal type: {metal_type}")
        
        # Use Replicate for background removal and enhancement
        client = replicate.Client(api_token=os.environ.get('REPLICATE_API_TOKEN'))
        
        # Convert to base64 for Replicate
        _, buffer = cv2.imencode('.png', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        try:
            # Use FLUX Fill for high-quality inpainting
            output = client.run(
                "black-forest-labs/flux-fill-dev",
                input={
                    "prompt": f"professional product photography of {metal_type.replace('_', ' ')} wedding ring, centered, white background, high detail, sharp focus",
                    "image": f"data:image/png;base64,{img_base64}",
                    "aspect_ratio": "4:5",  # Close to 1000x1300
                    "output_format": "png",
                    "output_quality": 100
                }
            )
            
            # Download result
            if output and isinstance(output, list) and len(output) > 0:
                response = requests.get(output[0])
                if response.status_code == 200:
                    img = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
            else:
                print("Replicate didn't return expected output, using original")
        except Exception as e:
            print(f"Replicate processing failed: {e}")
        
        # Find ring precisely
        x, y, w, h = find_ring_precise(img)
        
        # Add minimal padding (to avoid cutting edges)
        pad = 20
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img.shape[1], x + w + pad)
        y2 = min(img.shape[0], y + h + pad)
        
        # Crop ring region
        ring_crop = img[y1:y2, x1:x2]
        
        # Enhance ring details
        ring_crop = enhance_ring_details(ring_crop, metal_type)
        
        # Target size
        target_w, target_h = 1000, 1300
        
        # Calculate scale to maximize ring size
        crop_h, crop_w = ring_crop.shape[:2]
        scale = min(target_w/crop_w, target_h/crop_h) * 0.95  # 95% to leave tiny margin
        
        new_w = int(crop_w * scale)
        new_h = int(crop_h * scale)
        
        # Resize with high quality
        resized = cv2.resize(ring_crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Apply final sharpening
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        resized = cv2.filter2D(resized, -1, kernel)
        
        # Create white background
        thumbnail = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
        
        # Position ring (slightly higher for better composition)
        y_offset = max(10, (target_h - new_h) // 3)  # Top third
        x_offset = (target_w - new_w) // 2
        
        # Ensure we don't go out of bounds
        y_end = min(y_offset + new_h, target_h)
        x_end = min(x_offset + new_w, target_w)
        
        thumbnail[y_offset:y_end, x_offset:x_end] = resized[:y_end-y_offset, :x_end-x_offset]
        
        # Final brightness/contrast adjustment
        thumbnail = cv2.convertScaleAbs(thumbnail, alpha=1.1, beta=5)
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 95])
        thumb_base64 = remove_padding_safe(base64.b64encode(buffer).decode('utf-8'))
        
        return {
            "output": {
                "thumbnail": thumb_base64,
                "processing_info": {
                    "metal_type": metal_type,
                    "crop_info": {
                        "original_bbox": [x, y, w, h],
                        "final_size": [new_w, new_h],
                        "scale_factor": scale
                    },
                    "status": "success",
                    "version": "v1.0"
                }
            }
        }
        
    except Exception as e:
        print(f"Error in thumbnail handler: {str(e)}")
        return {
            "output": {
                "error": str(e),
                "status": "error"
            }
        }
