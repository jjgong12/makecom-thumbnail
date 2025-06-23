import os
import io
import json
import base64
import runpod
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import replicate
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import time

VERSION = "v4"

def log_debug(message):
    """Debug logging with version info"""
    print(f"[{VERSION}] {message}")

def find_image_in_event(event):
    """Enhanced image finding with multiple strategies to prevent Exit Code 0"""
    try:
        log_debug("Starting enhanced image search")
        
        # Strategy 1: Direct input keys
        job_input = event.get("input", {})
        log_debug(f"Event keys: {list(event.keys())}")
        log_debug(f"Input keys: {list(job_input.keys())}")
        
        # Check common image keys
        possible_keys = ["image", "image_base64", "base64_image", "data", "img", "file"]
        
        for key in possible_keys:
            if key in job_input and job_input[key]:
                log_debug(f"Found image data in key: {key}")
                return job_input[key]
        
        # Strategy 2: Direct event keys (fallback)
        for key in possible_keys:
            if key in event and event[key]:
                log_debug(f"Found image data in event key: {key}")
                return event[key]
        
        # Strategy 3: String input (direct base64)
        if isinstance(job_input, str) and len(job_input) > 100:
            log_debug("Found string input as potential base64")
            return job_input
        
        # Strategy 4: Nested search
        for key, value in job_input.items():
            if isinstance(value, dict):
                for nested_key in possible_keys:
                    if nested_key in value and value[nested_key]:
                        log_debug(f"Found nested image data in: {key}.{nested_key}")
                        return value[nested_key]
            elif isinstance(value, str) and len(value) > 100:
                log_debug(f"Found potential base64 in key: {key}")
                return value
        
        log_debug("No image data found in any strategy")
        return None
        
    except Exception as e:
        log_debug(f"Error in image search: {e}")
        return None

def decode_base64_image(base64_string):
    """Decode base64 image with enhanced error handling"""
    try:
        log_debug("Starting base64 decode process")
        
        # Remove whitespace and newlines
        base64_string = base64_string.strip().replace('\n', '').replace('\r', '').replace(' ', '')
        
        # Remove data URL prefix if present
        if 'base64,' in base64_string:
            base64_string = base64_string.split('base64,')[1]
            log_debug("Removed data URL prefix")
        
        # Clean any non-base64 characters
        import re
        base64_string = re.sub(r'[^A-Za-z0-9+/=]', '', base64_string)
        
        log_debug(f"Base64 string length after cleaning: {len(base64_string)}")
        
        # Try standard decode first (without adding padding)
        try:
            image_data = base64.b64decode(base64_string, validate=True)
            log_debug("Standard decode successful")
        except Exception as e:
            log_debug(f"Standard decode failed: {e}, trying with padding")
            # Add padding if needed
            missing_padding = len(base64_string) % 4
            if missing_padding:
                base64_string += '=' * (4 - missing_padding)
            image_data = base64.b64decode(base64_string, validate=True)
            log_debug("Decode with padding successful")
        
        # Convert to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image data")
            
        log_debug(f"Image decoded successfully: {image.shape}")
        return image
        
    except Exception as e:
        log_debug(f"Base64 decode error: {e}")
        raise

def detect_black_rectangle_masking(image):
    """Detect black rectangle masking in 6720x4480 image using comprehensive detection"""
    try:
        log_debug("Starting black rectangle masking detection for high-res image")
        
        h, w = image.shape[:2]
        log_debug(f"Image dimensions: {w}x{h}")
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        best_mask = None
        best_score = 0
        
        # Strategy 1: Edge-based detection (traditional)
        log_debug("Strategy 1: Edge-based detection")
        edge_masks = detect_edge_based_masking(gray, w, h)
        
        # Strategy 2: Center-based detection (for Image 1 style)
        log_debug("Strategy 2: Center-based detection")
        center_masks = detect_center_based_masking(gray, w, h)
        
        # Strategy 3: Contour-based detection
        log_debug("Strategy 3: Contour-based detection")
        contour_masks = detect_contour_based_masking(gray, w, h)
        
        # Combine all detected masks
        all_masks = edge_masks + center_masks + contour_masks
        
        # Score and select best mask
        for mask_info in all_masks:
            score = calculate_mask_score(mask_info, w, h)
            if score > best_score:
                best_score = score
                best_mask = mask_info
        
        if best_mask and best_score > 0.3:  # Minimum confidence threshold
            log_debug(f"Best mask found: {best_mask['type']} with score {best_score:.3f}")
            return best_mask
        else:
            log_debug("No valid masking detected")
            return None
            
    except Exception as e:
        log_debug(f"Masking detection error: {e}")
        return None

def detect_edge_based_masking(gray, w, h):
    """Detect masking that touches image edges"""
    masks = []
    
    # Multiple threshold values for different lighting conditions
    thresholds = [20, 30, 40, 50, 60]
    scan_percentages = [0.02, 0.05, 0.10, 0.15, 0.20]
    
    for threshold in thresholds:
        for scan_pct in scan_percentages:
            scan_depth = int(min(w, h) * scan_pct)
            
            # Scan from all four edges
            top_scan = np.mean(gray[:scan_depth, :]) < threshold
            bottom_scan = np.mean(gray[-scan_depth:, :]) < threshold
            left_scan = np.mean(gray[:, :scan_depth]) < threshold
            right_scan = np.mean(gray[:, -scan_depth:]) < threshold
            
            edge_count = sum([top_scan, bottom_scan, left_scan, right_scan])
            
            if edge_count >= 2:  # At least 2 edges touch
                # Find precise boundaries
                mask_info = find_precise_boundaries(gray, threshold, w, h)
                if mask_info:
                    mask_info['type'] = 'edge_based'
                    mask_info['edge_count'] = edge_count
                    masks.append(mask_info)
    
    return masks

def detect_center_based_masking(gray, w, h):
    """Detect masking in center area (like Image 1)"""
    masks = []
    
    # Focus on center 80% of image
    center_margin_w = int(w * 0.1)
    center_margin_h = int(h * 0.1)
    center_region = gray[center_margin_h:-center_margin_h, center_margin_w:-center_margin_w]
    
    # Multiple threshold values
    thresholds = [15, 25, 35, 45]
    
    for threshold in thresholds:
        # Find dark rectangular regions
        _, binary = cv2.threshold(center_region, threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Get bounding rectangle
            x, y, cw, ch = cv2.boundingRect(contour)
            
            # Convert back to full image coordinates
            full_x = x + center_margin_w
            full_y = y + center_margin_h
            
            # Check if it's a reasonable size (5% to 40% of image)
            area_ratio = (cw * ch) / (w * h)
            if 0.05 <= area_ratio <= 0.4:
                # Check aspect ratio (should be rectangular)
                aspect_ratio = max(cw, ch) / min(cw, ch)
                if 1.2 <= aspect_ratio <= 5.0:
                    mask_info = {
                        'x': full_x,
                        'y': full_y,
                        'w': cw,
                        'h': ch,
                        'type': 'center_rectangle',
                        'area_ratio': area_ratio,
                        'aspect_ratio': aspect_ratio
                    }
                    masks.append(mask_info)
    
    return masks

def detect_contour_based_masking(gray, w, h):
    """Detect masking using contour analysis"""
    masks = []
    
    # Apply adaptive threshold
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Get bounding rectangle
        x, y, cw, ch = cv2.boundingRect(contour)
        
        # Check size constraints
        area_ratio = (cw * ch) / (w * h)
        if 0.03 <= area_ratio <= 0.5:
            # Approximate contour to check if it's rectangular
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # If roughly rectangular (4-8 points)
            if 4 <= len(approx) <= 8:
                mask_info = {
                    'x': x,
                    'y': y,
                    'w': cw,
                    'h': ch,
                    'type': 'contour_based',
                    'vertices': len(approx),
                    'area_ratio': area_ratio
                }
                masks.append(mask_info)
    
    return masks

def find_precise_boundaries(gray, threshold, w, h):
    """Find precise boundaries of detected masking"""
    try:
        # Create binary mask
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Find the largest dark region
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(largest_contour)
        
        # Validate size
        area_ratio = (cw * ch) / (w * h)
        if area_ratio < 0.02 or area_ratio > 0.6:
            return None
        
        return {
            'x': x,
            'y': y, 
            'w': cw,
            'h': ch,
            'area_ratio': area_ratio
        }
        
    except Exception as e:
        log_debug(f"Boundary detection error: {e}")
        return None

def calculate_mask_score(mask_info, img_w, img_h):
    """Calculate confidence score for detected mask"""
    score = 0.0
    
    # Size score (prefer 10-30% of image)
    area_ratio = mask_info['area_ratio']
    if 0.1 <= area_ratio <= 0.3:
        score += 0.4
    elif 0.05 <= area_ratio <= 0.5:
        score += 0.2
    
    # Type score (prefer center rectangles for this use case)
    if mask_info['type'] == 'center_rectangle':
        score += 0.3
    elif mask_info['type'] == 'edge_based':
        score += 0.2
    elif mask_info['type'] == 'contour_based':
        score += 0.15
    
    # Aspect ratio score
    if 'aspect_ratio' in mask_info:
        aspect = mask_info['aspect_ratio']
        if 1.0 <= aspect <= 2.0:  # Square to mild rectangle
            score += 0.2
        elif 2.0 <= aspect <= 4.0:  # Moderate rectangle
            score += 0.1
    
    # Edge count bonus (for edge-based)
    if 'edge_count' in mask_info and mask_info['edge_count'] >= 3:
        score += 0.1
    
    return score

def remove_masking_with_replicate(image, mask_info):
    """Remove masking using Replicate API"""
    try:
        log_debug("Starting Replicate API masking removal")
        
        # Create mask for inpainting
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        x, y, mw, mh = mask_info['x'], mask_info['y'], mask_info['w'], mask_info['h']
        
        # Add some padding to ensure complete removal
        padding = 10
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + mw + padding)
        y2 = min(h, y + mh + padding)
        
        mask[y1:y2, x1:x2] = 255
        
        # Convert to base64 for Replicate
        _, img_buffer = cv2.imencode('.png', image)
        img_base64 = base64.b64encode(img_buffer).decode()
        
        _, mask_buffer = cv2.imencode('.png', mask)
        mask_base64 = base64.b64encode(mask_buffer).decode()
        
        # Call Replicate API with timeout
        def run_replicate():
            client = replicate.Client(api_token=os.environ.get("REPLICATE_API_TOKEN"))
            
            # Use FLUX Fill for high quality inpainting
            output = client.run(
                "black-forest-labs/flux-fill-dev",
                input={
                    "image": f"data:image/png;base64,{img_base64}",
                    "mask": f"data:image/png;base64,{mask_base64}",
                    "prompt": "clean white seamless background",
                    "num_inference_steps": 25,
                    "guidance_scale": 7.5,
                    "strength": 0.95
                }
            )
            return output
        
        try:
            with ThreadPoolExecutor() as executor:
                future = executor.submit(run_replicate)
                result = future.result(timeout=30)
                
            if result:
                log_debug("Replicate API masking removal successful")
                # Download and convert result
                import requests
                response = requests.get(result)
                result_array = np.frombuffer(response.content, np.uint8)
                cleaned_image = cv2.imdecode(result_array, cv2.IMREAD_COLOR)
                return cleaned_image
                
        except TimeoutError:
            log_debug("Replicate API timeout, using fallback")
        except Exception as e:
            log_debug(f"Replicate API error: {e}, using fallback")
        
        # Fallback: simple inpainting
        return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        
    except Exception as e:
        log_debug(f"Masking removal error: {e}")
        return image

def extract_ring_region_and_create_thumbnail(image, mask_info):
    """Extract ring region from inside the mask and create 1000x1300 thumbnail"""
    try:
        log_debug("Extracting ring region for thumbnail creation")
        
        x, y, w, h = mask_info['x'], mask_info['y'], mask_info['w'], mask_info['h']
        
        # Extract the region inside the mask (where the rings are)
        ring_region = image[y:y+h, x:x+w].copy()
        log_debug(f"Ring region extracted: {ring_region.shape}")
        
        # Apply Enhancement Handler style color correction
        enhanced_ring = apply_enhancement_color_correction(ring_region)
        
        # Apply light detail enhancement specifically for rings
        detail_enhanced = apply_ring_detail_enhancement(enhanced_ring)
        
        # Create 1000x1300 thumbnail with ring centered
        thumbnail = create_centered_thumbnail(detail_enhanced, (1000, 1300))
        
        log_debug("Ring thumbnail created successfully")
        return thumbnail
        
    except Exception as e:
        log_debug(f"Ring extraction error: {e}")
        # Fallback: use center region
        h, w = image.shape[:2]
        center_region = image[h//4:3*h//4, w//4:3*w//4]
        return cv2.resize(center_region, (1000, 1300), interpolation=cv2.INTER_LANCZOS4)

def apply_enhancement_color_correction(image):
    """Apply same color correction as Enhancement Handler"""
    try:
        log_debug("Applying Enhancement Handler color correction to ring")
        
        # Convert to RGB for PIL processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Enhancement Handler style corrections
        brightness_enhancer = ImageEnhance.Brightness(pil_image)
        enhanced = brightness_enhancer.enhance(1.25)  # 25% brightness boost
        
        contrast_enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = contrast_enhancer.enhance(1.1)  # 10% contrast
        
        color_enhancer = ImageEnhance.Color(enhanced)
        enhanced = color_enhancer.enhance(0.95)  # 5% saturation reduction
        
        # Convert back to numpy
        enhanced_array = np.array(enhanced)
        enhanced_bgr = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2BGR)
        
        # 15% white overlay
        white_overlay = np.full_like(enhanced_bgr, 255, dtype=np.uint8)
        enhanced_bgr = cv2.addWeighted(enhanced_bgr, 0.85, white_overlay, 0.15, 0)
        
        return enhanced_bgr
        
    except Exception as e:
        log_debug(f"Color correction error: {e}")
        return image

def apply_ring_detail_enhancement(image):
    """Apply light detail enhancement for ring details"""
    try:
        log_debug("Applying ring detail enhancement")
        
        # Light noise reduction
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 3, 3, 7, 21)
        
        # Ring-specific sharpening (stronger than general enhancement)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        # Light CLAHE for local contrast enhancement
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Additional brightness boost for ring details
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=5)
        
        log_debug("Ring detail enhancement completed")
        return enhanced
        
    except Exception as e:
        log_debug(f"Detail enhancement error: {e}")
        return image

def create_centered_thumbnail(ring_image, target_size=(1000, 1300)):
    """Create perfectly centered 1000x1300 thumbnail"""
    try:
        log_debug(f"Creating centered thumbnail {target_size}")
        
        target_w, target_h = target_size
        ring_h, ring_w = ring_image.shape[:2]
        
        # Calculate scaling to maximize ring size while maintaining aspect ratio
        scale_x = target_w / ring_w
        scale_y = target_h / ring_h
        scale = min(scale_x, scale_y) * 0.9  # Use 90% to leave some margin
        
        # Resize ring
        new_w = int(ring_w * scale)
        new_h = int(ring_h * scale)
        resized_ring = cv2.resize(ring_image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Create white background
        thumbnail = np.ones((target_h, target_w, 3), dtype=np.uint8) * 250  # Slightly off-white
        
        # Center the ring
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        # Place ring in center
        thumbnail[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_ring
        
        log_debug(f"Thumbnail created: ring size {new_w}x{new_h} in {target_w}x{target_h}")
        return thumbnail
        
    except Exception as e:
        log_debug(f"Thumbnail creation error: {e}")
        return cv2.resize(ring_image, target_size, interpolation=cv2.INTER_LANCZOS4)

def image_to_base64(image):
    """Convert image to base64 string without padding (Make.com compatible)"""
    try:
        _, buffer = cv2.imencode('.png', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        # CRITICAL: Remove padding for Make.com compatibility
        img_base64 = img_base64.rstrip('=')
        log_debug("Image converted to base64 (padding removed)")
        return img_base64
    except Exception as e:
        log_debug(f"Base64 conversion error: {e}")
        raise

def handler(event):
    """Main handler function for thumbnail generation with masking"""
    try:
        log_debug("=== Thumbnail Handler v4 Started ===")
        log_debug("Black rectangle masking detection for 6720x4480 images")
        
        # Enhanced image finding to prevent Exit Code 0
        base64_image = find_image_in_event(event)
        
        if not base64_image:
            log_debug("No image provided - returning error response")
            return {
                "output": {
                    "error": "No image provided",
                    "status": "error", 
                    "version": VERSION,
                    "timestamp": int(time.time())
                }
            }
        
        log_debug(f"Base64 string length: {len(base64_image)}")
        
        # Decode image
        image = decode_base64_image(base64_image)
        log_debug(f"Image decoded: {image.shape}")
        
        # Detect black rectangle masking
        mask_info = detect_black_rectangle_masking(image)
        
        if mask_info:
            log_debug(f"Masking detected: {mask_info}")
            
            # Remove masking using Replicate API
            cleaned_image = remove_masking_with_replicate(image, mask_info)
            
            # Extract ring region and create thumbnail
            thumbnail = extract_ring_region_and_create_thumbnail(cleaned_image, mask_info)
            
        else:
            log_debug("No masking detected, using center region")
            # Fallback: use center region with enhancement
            h, w = image.shape[:2]
            center_region = image[h//4:3*h//4, w//4:3*w//4]
            enhanced_center = apply_enhancement_color_correction(center_region)
            detail_enhanced = apply_ring_detail_enhancement(enhanced_center)
            thumbnail = create_centered_thumbnail(detail_enhanced, (1000, 1300))
        
        # Convert to base64
        thumbnail_base64 = image_to_base64(thumbnail)
        
        # Prepare response
        response = {
            "output": {
                "thumbnail": thumbnail_base64,
                "processing_info": {
                    "version": VERSION,
                    "masking_detected": mask_info is not None,
                    "masking_type": mask_info['type'] if mask_info else "none",
                    "thumbnail_size": "1000x1300",
                    "enhancement_applied": True,
                    "detail_enhancement": True,
                    "timestamp": int(time.time())
                }
            }
        }
        
        log_debug("=== Thumbnail Handler v4 Completed Successfully ===")
        return response
        
    except Exception as e:
        log_debug(f"Handler error: {e}")
        
        # Always return proper response structure to prevent Exit Code 0
        return {
            "output": {
                "error": str(e),
                "status": "error",
                "version": VERSION,
                "timestamp": int(time.time())
            }
        }

# RunPod serverless handler
if __name__ == "__main__":
    log_debug("Starting RunPod serverless handler")
    runpod.serverless.start({"handler": handler})
