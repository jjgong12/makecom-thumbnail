import runpod
import base64
import requests
from io import BytesIO
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import numpy as np
import cv2
import os
import time
import json
import traceback
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VERSION = "Thumbnail_V67_COLOR_PRESERVE"

# V67: Get API token from environment variable FIRST
REPLICATE_API_TOKEN = os.environ.get('REPLICATE_API_TOKEN', 'r8_6cksfxEmLxWlYxjW4K1FEbnZMEEmlQw2UeNNY')

def create_session():
    """Create a session with retry strategy"""
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def find_input_data(data):
    """Recursively find input data from various possible locations"""
    logger.info("Searching for input data...")
    
    # Direct access attempts
    if isinstance(data, dict):
        # Check top level
        if 'input' in data:
            return data['input']
        
        # Common RunPod structures
        common_paths = [
            ['job', 'input'],
            ['data', 'input'],
            ['payload', 'input'],
            ['body', 'input'],
            ['request', 'input']
        ]
        
        for path in common_paths:
            current = data
            for key in path:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    break
            else:
                logger.info(f"Found input at path: {'.'.join(path)}")
                return current
    
    # Recursive search function
    def recursive_search(obj, target_keys=None):
        if target_keys is None:
            target_keys = ['input', 'url', 'image_url', 'imageUrl', 'image_base64', 
                          'imageBase64', 'image', 'enhanced_image', 'base64_image']
        
        if isinstance(obj, dict):
            # Check for target keys
            for key in target_keys:
                if key in obj:
                    value = obj[key]
                    if key == 'input':
                        return value
                    else:
                        # Return as dict to maintain structure
                        return {key: value}
            
            # Recursive search in values
            for value in obj.values():
                result = recursive_search(value, target_keys)
                if result:
                    return result
                    
        elif isinstance(obj, list):
            for item in obj:
                result = recursive_search(item, target_keys)
                if result:
                    return result
        
        return None
    
    # Try recursive search
    result = recursive_search(data)
    if result:
        logger.info(f"Found input via recursive search: {type(result)}")
        return result
    
    # Last resort - check if the data itself is the input
    if isinstance(data, str) and len(data) > 100:
        logger.info("Using raw data as input")
        return data
    
    logger.warning("No input data found")
    return None

def decode_base64_safe(base64_str):
    """Decode base64 with automatic padding correction"""
    try:
        # Clean the string
        if 'base64,' in base64_str:
            base64_str = base64_str.split('base64,')[1]
        
        # Fix padding if needed
        padding = 4 - len(base64_str) % 4
        if padding != 4:
            base64_str += '=' * padding
        
        return base64.b64decode(base64_str)
    except Exception as e:
        logger.error(f"Base64 decode error: {str(e)}")
        raise

def apply_gamma_correction(image, gamma):
    """Apply gamma correction to brighten mid-tones"""
    try:
        img_array = np.array(image).astype(float) / 255.0
        corrected = np.power(img_array, gamma)
        corrected = (corrected * 255).astype(np.uint8)
        return Image.fromarray(corrected)
    except Exception as e:
        logger.error(f"Gamma correction error: {str(e)}")
        return image

def apply_clahe_enhancement(image):
    """Apply CLAHE for micro-contrast enhancement"""
    try:
        img_array = np.array(image)
        
        # Convert to LAB
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge and convert back
        enhanced_lab = cv2.merge([l, a, b])
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        return Image.fromarray(enhanced_rgb)
    except Exception as e:
        logger.error(f"CLAHE error: {str(e)}")
        return image

def apply_super_resolution_enhance(image):
    """Apply super-resolution-like enhancement"""
    try:
        img_np = np.array(image)
        
        # 1. Sharpen
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]]) / 1.0
        sharpened = cv2.filter2D(img_np, -1, kernel)
        
        # 2. Edge enhancement
        edges = cv2.Canny(cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY), 50, 150)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        # 3. Blend
        result = cv2.addWeighted(sharpened, 0.8, edges_colored, 0.05, 0)
        
        # 4. Denoise
        result = cv2.fastNlMeansDenoisingColored(result, None, 5, 5, 7, 21)
        
        return Image.fromarray(result)
    except Exception as e:
        logger.error(f"Super resolution error: {str(e)}")
        return image

def detect_ring_color_improved(image):
    """Improved color detection that preserves original colors"""
    try:
        # Convert to numpy array
        img_array = np.array(image)
        
        # Get center region (where ring is likely to be)
        h, w = img_array.shape[:2]
        center_y, center_x = h // 2, w // 2
        region_size = min(h, w) // 3
        
        center_region = img_array[
            center_y - region_size:center_y + region_size,
            center_x - region_size:center_x + region_size
        ]
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(center_region, cv2.COLOR_RGB2HSV)
        
        # Calculate color statistics
        avg_h = np.mean(hsv[:, :, 0])
        avg_s = np.mean(hsv[:, :, 1])
        avg_v = np.mean(hsv[:, :, 2])
        
        # Also check LAB color space
        lab = cv2.cvtColor(center_region, cv2.COLOR_RGB2LAB)
        avg_l = np.mean(lab[:, :, 0])
        avg_a = np.mean(lab[:, :, 1])
        avg_b = np.mean(lab[:, :, 2])
        
        # RGB average for better yellow detection
        avg_r = np.mean(center_region[:, :, 0])
        avg_g = np.mean(center_region[:, :, 1])
        avg_b_rgb = np.mean(center_region[:, :, 2])
        
        logger.info(f"Color analysis - H: {avg_h:.1f}, S: {avg_s:.1f}, V: {avg_v:.1f}")
        logger.info(f"LAB analysis - L: {avg_l:.1f}, a: {avg_a:.1f}, b: {avg_b:.1f}")
        logger.info(f"RGB analysis - R: {avg_r:.1f}, G: {avg_g:.1f}, B: {avg_b_rgb:.1f}")
        
        # V67: Improved color detection logic
        # Yellow gold detection - warm tones
        if (15 <= avg_h <= 40 and avg_s > 25 and avg_b > 140) or \
           (avg_r > avg_b_rgb + 20 and avg_g > avg_b_rgb + 10):
            return 'yellow_gold'
        # Rose gold detection - pink/reddish tones
        elif ((avg_h < 15 or avg_h > 165) and avg_s > 20 and avg_a > 135) or \
             (avg_r > avg_g + 5 and avg_r > avg_b_rgb + 10):
            return 'rose_gold'
        # White detection - very bright and desaturated
        elif avg_v > 200 and avg_s < 25 and abs(avg_a - 128) < 10:
            return 'white'
        # White gold - bright but slightly cool
        elif avg_s < 40 and avg_v > 150 and avg_b_rgb >= avg_r:
            return 'white_gold'
        # Default to yellow gold for warm tones
        elif avg_r >= avg_b_rgb:
            return 'yellow_gold'
        else:
            return 'white_gold'
            
    except Exception as e:
        logger.error(f"Color detection error: {str(e)}")
        return 'yellow_gold'  # Default to yellow gold

def apply_color_specific_enhancement(image, color):
    """Apply color-specific enhancements while preserving original colors"""
    try:
        # Ensure RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        logger.info(f"Applying {color}-specific enhancement...")
        
        if color == 'yellow_gold':
            # Preserve warm tones
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.1)  # Slight brightness
            
            # Enhance gold tone slightly
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(1.1)  # Slight saturation boost
            
            # Mild contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.05)
            
            # Warm gamma
            image = apply_gamma_correction(image, 0.9)
            
        elif color == 'rose_gold':
            # Preserve pink tones
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.08)
            
            # Maintain color
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(1.05)
            
            # Soft contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.05)
            
            # Preserve warmth
            image = apply_gamma_correction(image, 0.85)
            
        elif color == 'white':
            # V67: Pure white enhancement
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.35)
            
            # Strong desaturation
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(0.5)
            
            # High contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.15)
            
            # Bright gamma
            image = apply_gamma_correction(image, 0.7)
            
        elif color == 'white_gold':
            # Cool metallic
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.15)
            
            # Slight desaturation
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(0.8)
            
            # Moderate contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.1)
            
            image = apply_gamma_correction(image, 0.8)
        
        # Apply CLAHE for all colors
        image = apply_clahe_enhancement(image)
        
        return image
        
    except Exception as e:
        logger.error(f"Enhancement error: {str(e)}")
        return image

def detect_wedding_rings(image):
    """Detect wedding rings in image for better cropping"""
    try:
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Detect circles using HoughCircles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=20,
            maxRadius=200
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            logger.info(f"Detected {len(circles[0])} potential rings")
            
            # Find the center point of all detected circles
            if len(circles[0]) > 0:
                center_x = int(np.mean([c[0] for c in circles[0]]))
                center_y = int(np.mean([c[1] for c in circles[0]]))
                return (center_x, center_y)
        
        # If no circles found, return image center
        h, w = img_array.shape[:2]
        return (w // 2, h // 2)
        
    except Exception as e:
        logger.error(f"Ring detection error: {str(e)}")
        h, w = np.array(image).shape[:2]
        return (w // 2, h // 2)

def create_thumbnail_with_less_zoom(image, target_size=(1000, 1300)):
    """Create thumbnail with less zoom - V67: 55% crop"""
    try:
        # First, detect wedding rings to find optimal crop center
        ring_center = detect_wedding_rings(image)
        logger.info(f"Ring center detected at: {ring_center}")
        
        # Calculate crop dimensions with less zoom
        target_ratio = target_size[0] / target_size[1]  # 1000/1300 = 0.769
        img_width, img_height = image.size
        
        # V67: Use 55% of the image
        crop_percentage = 0.55
        
        # Determine crop size
        if img_width / img_height > target_ratio:
            # Image is wider - crop width
            crop_height = int(img_height * crop_percentage)
            crop_width = int(crop_height * target_ratio)
        else:
            # Image is taller - crop height
            crop_width = int(img_width * crop_percentage)
            crop_height = int(crop_width / target_ratio)
        
        # Center crop around detected ring center
        left = max(0, ring_center[0] - crop_width // 2)
        top = max(0, ring_center[1] - crop_height // 2)
        
        # Adjust if crop goes out of bounds
        if left + crop_width > img_width:
            left = img_width - crop_width
        if top + crop_height > img_height:
            top = img_height - crop_height
        
        logger.info(f"Crop area: ({left}, {top}) size {crop_width}x{crop_height}")
        
        # Perform crop
        cropped = image.crop((left, top, left + crop_width, top + crop_height))
        
        # Resize to exact target size
        thumbnail = cropped.resize(target_size, Image.Resampling.LANCZOS)
        
        return thumbnail
        
    except Exception as e:
        logger.error(f"Crop error: {str(e)}")
        # Fallback to simple center crop
        return image.resize(target_size, Image.Resampling.LANCZOS)

def enhance_with_replicate_color_preserve(image_base64, color):
    """Use Replicate API for quality enhancement with color preservation"""
    try:
        # V67: Skip Replicate for yellow/rose gold to preserve color
        if color in ['yellow_gold', 'rose_gold']:
            logger.info(f"Skipping Replicate for {color} to preserve color")
            return None
            
        logger.info("Starting Replicate enhancement...")
        session = create_session()
        
        # Remove padding for Replicate
        image_base64_clean = image_base64.rstrip('=')
        
        headers = {
            "Authorization": f"Token {REPLICATE_API_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # Use Real-ESRGAN for quality enhancement
        logger.info("Creating prediction with Real-ESRGAN...")
        create_response = session.post(
            "https://api.replicate.com/v1/predictions",
            json={
                "version": "42fed1c4974146d4d2414e2be2c5277c7fcf05fcc3a73abf41610695738c1d7b",
                "input": {
                    "image": f"data:image/png;base64,{image_base64_clean}",
                    "scale": 2,
                    "face_enhance": False
                }
            },
            headers=headers,
            timeout=30
        )
        
        if create_response.status_code != 201:
            logger.error(f"Failed to create prediction: {create_response.status_code}")
            return None
            
        prediction = create_response.json()
        prediction_id = prediction['id']
        logger.info(f"Prediction ID: {prediction_id}")
        
        # Poll for result
        max_attempts = 30
        for attempt in range(max_attempts):
            time.sleep(1)
            
            get_response = session.get(
                f"https://api.replicate.com/v1/predictions/{prediction_id}",
                headers=headers,
                timeout=30
            )
            
            if get_response.status_code != 200:
                logger.error(f"Failed to get prediction status: {get_response.status_code}")
                continue
                
            result = get_response.json()
            status = result.get('status', '')
            
            logger.info(f"Attempt {attempt + 1}/{max_attempts}: Status = {status}")
            
            if status == 'succeeded':
                output_url = result.get('output')
                if not output_url:
                    logger.error("No output URL in result")
                    return None
                
                logger.info(f"Downloading enhanced image...")
                
                # Download result image
                img_response = session.get(output_url, timeout=30)
                if img_response.status_code == 200:
                    # Load enhanced image
                    enhanced_img = Image.open(BytesIO(img_response.content))
                    
                    # Convert back to base64
                    buffered = BytesIO()
                    enhanced_img.save(buffered, format="PNG", optimize=True)
                    result_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    
                    logger.info("Replicate enhancement successful")
                    return result_base64
                else:
                    logger.error(f"Failed to download result: {img_response.status_code}")
                    return None
                    
            elif status == 'failed':
                error = result.get('error', 'Unknown error')
                logger.error(f"Prediction failed: {error}")
                return None
        
        logger.info("Timeout waiting for prediction")
        return None
        
    except Exception as e:
        logger.error(f"Replicate API error: {str(e)}")
        return None

def apply_final_color_enhancement(image, color):
    """Apply final color-specific enhancements"""
    try:
        enhanced = image.copy()
        
        if color == 'white':
            # Maximum white enhancement
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(1.2)
            
            enhancer = ImageEnhance.Color(enhanced)
            enhanced = enhancer.enhance(0.4)
            
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(1.2)
            
        elif color == 'yellow_gold':
            # Preserve gold warmth
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(1.05)
            
            enhancer = ImageEnhance.Color(enhanced)
            enhanced = enhancer.enhance(1.05)
            
        elif color == 'rose_gold':
            # Preserve pink tones
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(1.05)
            
        elif color == 'white_gold':
            # Cool metallic
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(1.1)
            
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(1.1)
        
        # Moderate sharpness for all
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(1.5)
        
        # Final super-resolution enhancement
        enhanced = apply_super_resolution_enhance(enhanced)
        
        return enhanced
        
    except Exception as e:
        logger.error(f"Final enhancement error: {str(e)}")
        return image

def process_thumbnail(job):
    """Process thumbnail request with v67 color preservation"""
    logger.info(f"=== {VERSION} Started ===")
    logger.info(f"Received job: {json.dumps(job, indent=2)[:500]}...")
    
    start_time = time.time()
    
    try:
        # Extract input using the job parameter correctly
        job_input = job.get('input', {})
        
        # Find image data
        input_data = find_input_data(job_input)
        
        # If not found in job_input, try the whole job dict
        if not input_data:
            input_data = find_input_data(job)
        
        if not input_data:
            error_msg = "No image data provided in any expected field"
            logger.error(error_msg)
            return {
                "output": {
                    "error": error_msg,
                    "status": "error",
                    "available_fields": list(job_input.keys()) if isinstance(job_input, dict) else []
                }
            }
        
        # Extract parameters
        image_data = None
        
        if isinstance(input_data, dict):
            # Extract image
            for key in ['image', 'image_base64', 'imageBase64', 'base64_image', 
                       'url', 'image_url', 'imageUrl', 'enhanced_image']:
                if key in input_data:
                    image_data = input_data[key]
                    break
        elif isinstance(input_data, str):
            image_data = input_data
        
        if not image_data:
            return {
                "output": {
                    "error": "Could not extract image data from input",
                    "status": "error"
                }
            }
        
        # Process image
        logger.info("Processing base64 image...")
        
        # Decode image
        image_bytes = decode_base64_safe(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        logger.info(f"Original image: {image.mode} {image.size}")
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            logger.info(f"Converting from {image.mode} to RGB")
            image = image.convert('RGB')
        
        # Step 1: Detect color FIRST before any processing
        logger.info("Step 1: Detecting ring color...")
        detected_color = detect_ring_color_improved(image)
        logger.info(f"Detected color: {detected_color}")
        
        # Step 2: Apply color-specific enhancement
        logger.info(f"Step 2: Applying {detected_color}-specific enhancement...")
        enhanced_image = apply_color_specific_enhancement(image, detected_color)
        
        # Step 3: Create 1000x1300 thumbnail with less zoom
        logger.info("Step 3: Creating thumbnail with 55% crop...")
        thumbnail = create_thumbnail_with_less_zoom(enhanced_image, (1000, 1300))
        
        # Step 4: Optional Replicate enhancement (skip for yellow/rose gold)
        logger.info("Step 4: Considering Replicate enhancement...")
        
        # Convert thumbnail to base64 for Replicate
        buffered = BytesIO()
        thumbnail.save(buffered, format="PNG")
        thumbnail_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Try Replicate enhancement with color preservation
        enhanced_base64 = enhance_with_replicate_color_preserve(thumbnail_base64, detected_color)
        
        if enhanced_base64:
            # Load enhanced image
            enhanced_bytes = decode_base64_safe(enhanced_base64)
            thumbnail = Image.open(BytesIO(enhanced_bytes))
            
            # Resize back to 1000x1300 if needed
            if thumbnail.size != (1000, 1300):
                thumbnail = thumbnail.resize((1000, 1300), Image.Resampling.LANCZOS)
        else:
            logger.info(f"Continuing with local processing for {detected_color}")
        
        # Step 5: Apply final color-specific enhancements
        logger.info("Step 5: Applying final enhancements...")
        final_thumbnail = apply_final_color_enhancement(thumbnail, detected_color)
        
        # Convert to base64
        final_buffer = BytesIO()
        final_thumbnail.save(final_buffer, format="PNG", optimize=True)
        final_base64 = base64.b64encode(final_buffer.getvalue()).decode('utf-8')
        
        # Prepare results with different padding options
        # For Make.com - no padding
        result_base64_no_padding = final_base64.rstrip('=')
        
        # For Google Script - with padding
        result_base64_with_padding = final_base64
        padding_needed = 4 - (len(final_base64) % 4)
        if padding_needed and padding_needed != 4:
            result_base64_with_padding = final_base64 + ('=' * padding_needed)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        logger.info(f"Thumbnail processing completed in {processing_time:.2f}s")
        
        # Color mapping
        color_map = {
            'yellow_gold': '#FFD700',
            'rose_gold': '#E8B4B8',
            'white_gold': '#F5F5F5',
            'white': '#FFFFFF'
        }
        
        return {
            "output": {
                "thumbnail": result_base64_no_padding,  # Make.com용 (padding 없음)
                "thumbnail_with_padding": result_base64_with_padding,  # Google Script용
                "color": color_map.get(detected_color, '#FFD700'),
                "status": "success",
                "message": f"Thumbnail created with v67 color preservation",
                "processing_time": f"{processing_time:.2f}s",
                "detected_color": detected_color,
                "original_size": list(image.size),
                "final_size": [1000, 1300],
                "settings": {
                    "size": "1000x1300",
                    "zoom_level": "55%",
                    "enhancement": "v67_color_preserve",
                    "replicate_used": enhanced_base64 is not None,
                    "color_name": detected_color,
                    "color_hex": color_map.get(detected_color, '#FFD700')
                }
            }
        }
        
    except Exception as e:
        error_msg = f"Thumbnail processing failed: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        return {
            "output": {
                "error": error_msg,
                "status": "error",
                "traceback": traceback.format_exc()
            }
        }

# RunPod handler
logger.info(f"Starting RunPod {VERSION}...")
runpod.serverless.start({"handler": process_thumbnail})
