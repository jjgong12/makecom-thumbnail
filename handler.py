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

VERSION = "Thumbnail_V64_ENV_PRIORITY"

# V64: Get API token from environment variable FIRST
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
    
    # Log the structure (limited to prevent huge logs)
    logger.info(f"Input structure: {json.dumps(data, indent=2)[:1000]}...")
    
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

def validate_base64(data):
    """Validate and clean base64 string"""
    try:
        # Remove data URL prefix if present
        if isinstance(data, str) and 'base64,' in data:
            data = data.split('base64,')[1]
        
        # Remove whitespace
        if isinstance(data, str):
            data = data.strip()
        
        # Try decoding
        base64.b64decode(data)
        return True, data
    except Exception as e:
        logger.error(f"Base64 validation error: {str(e)}")
        return False, None

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

def download_image_from_url(url):
    """Download image from URL"""
    try:
        session = create_session()
        response = session.get(url, timeout=30)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        logger.error(f"Failed to download image: {str(e)}")
        raise

def detect_wedding_rings(image):
    """Detect wedding rings in image"""
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

def center_crop_to_square(image):
    """Center crop image to square"""
    width, height = image.size
    size = min(width, height)
    
    # Calculate center crop coordinates
    left = (width - size) // 2
    top = (height - size) // 2
    right = left + size
    bottom = top + size
    
    return image.crop((left, top, right, bottom))

def create_thumbnail_with_crop(image, target_size=(1000, 1300)):
    """Create thumbnail with specific aspect ratio and size"""
    try:
        # First, detect wedding rings to find optimal crop center
        ring_center = detect_wedding_rings(image)
        logger.info(f"Ring center detected at: {ring_center}")
        
        # Calculate crop dimensions
        target_ratio = target_size[0] / target_size[1]  # 1000/1300 = 0.769
        img_width, img_height = image.size
        
        # Determine crop size
        if img_width / img_height > target_ratio:
            # Image is wider - crop width
            crop_height = img_height
            crop_width = int(crop_height * target_ratio)
        else:
            # Image is taller - crop height
            crop_width = img_width
            crop_height = int(crop_width / target_ratio)
        
        # Center crop around detected ring center
        left = max(0, ring_center[0] - crop_width // 2)
        top = max(0, ring_center[1] - crop_height // 2)
        
        # Adjust if crop goes out of bounds
        if left + crop_width > img_width:
            left = img_width - crop_width
        if top + crop_height > img_height:
            top = img_height - crop_height
        
        # Perform crop
        cropped = image.crop((left, top, left + crop_width, top + crop_height))
        
        # Resize to exact target size
        thumbnail = cropped.resize(target_size, Image.Resampling.LANCZOS)
        
        return thumbnail
        
    except Exception as e:
        logger.error(f"Crop error: {str(e)}")
        # Fallback to simple center crop
        return center_crop_to_square(image).resize(target_size, Image.Resampling.LANCZOS)

def detect_ring_color(image):
    """Detect ring color from image with improved accuracy"""
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
        
        logger.info(f"Color analysis - H: {avg_h:.1f}, S: {avg_s:.1f}, V: {avg_v:.1f}")
        logger.info(f"LAB analysis - L: {avg_l:.1f}, a: {avg_a:.1f}, b: {avg_b:.1f}")
        
        # V62: 무도금화이트 우선 감지
        if avg_v > 200 and avg_s < 30:  # Very bright and low saturation
            return 'white'
        elif avg_s < 40 and avg_v > 150 and avg_a < 130:  # Low saturation, bright, neutral
            return 'white_gold'
        elif 15 <= avg_h <= 35 and avg_s > 30 and avg_a > 130:  # Yellow hue with warmth
            return 'yellow_gold'
        elif (avg_h < 15 or avg_h > 165) and avg_s > 20 and avg_a > 135:  # Pink tone
            return 'rose_gold'
        else:
            return 'white'  # Default to white
            
    except Exception as e:
        logger.error(f"Color detection error: {str(e)}")
        return 'white'

def apply_color_specific_enhancement(image, color):
    """Apply color-specific enhancements"""
    try:
        enhanced = image.copy()
        
        if color == 'yellow_gold':
            # Yellow gold: warm enhancement
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(1.1)
            
            # Increase warmth
            img_array = np.array(enhanced)
            img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 0.95, 0, 255)  # Reduce blue
            img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 1.05, 0, 255)  # Increase red
            enhanced = Image.fromarray(img_array.astype(np.uint8))
            
        elif color == 'rose_gold':
            # Rose gold: pink tone enhancement
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(1.15)
            
            # Add pink tone
            img_array = np.array(enhanced)
            img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 0.9, 0, 255)   # Reduce blue
            img_array[:, :, 1] = np.clip(img_array[:, :, 1] * 0.95, 0, 255)  # Slightly reduce green
            enhanced = Image.fromarray(img_array.astype(np.uint8))
            
        elif color == 'white_gold':
            # White gold: cool metallic
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(1.15)
            
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(1.1)
            
            # Cool tone
            img_array = np.array(enhanced)
            img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 0.95, 0, 255)  # Reduce red
            enhanced = Image.fromarray(img_array.astype(np.uint8))
            
        else:  # white
            # Pure white: maximum brightness
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(1.25)
            
            # Reduce saturation
            enhancer = ImageEnhance.Color(enhanced)
            enhanced = enhancer.enhance(0.7)
        
        # Common enhancements for all
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(1.5)
        
        return enhanced
        
    except Exception as e:
        logger.error(f"Color enhancement error: {str(e)}")
        return image

def remove_background_with_replicate(image_base64, api_token):
    """Remove background using Replicate API with transparent background"""
    try:
        logger.info("Starting Replicate background removal...")
        session = create_session()
        
        # Remove padding for Replicate
        image_base64_clean = image_base64.rstrip('=')
        
        headers = {
            "Authorization": f"Token {api_token}",
            "Content-Type": "application/json"
        }
        
        # Create prediction - V62: 명시적으로 transparent 배경 설정
        logger.info("Creating prediction with transparent background...")
        create_response = session.post(
            "https://api.replicate.com/v1/predictions",
            json={
                "version": "3243b8f1cb654c8225867325394d3d60fb8284de3c212e87a1c6d0fc4c8203f6",
                "input": {
                    "image": f"data:image/png;base64,{image_base64_clean}",
                    "bg_color": "transparent"  # V62: 명시적으로 투명 배경 설정
                }
            },
            headers=headers,
            timeout=30
        )
        
        if create_response.status_code != 201:
            logger.error(f"Failed to create prediction: {create_response.status_code}")
            logger.error(f"Response: {create_response.text}")
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
                
                logger.info(f"Downloading result from: {output_url}")
                
                # Download result image
                img_response = session.get(output_url, timeout=30)
                if img_response.status_code == 200:
                    # V62: PNG로 변환하여 투명도 유지
                    img = Image.open(BytesIO(img_response.content))
                    
                    # V62: RGBA 모드 확인 (투명도 채널 필요)
                    if img.mode != 'RGBA':
                        logger.info(f"Converting from {img.mode} to RGBA")
                        img = img.convert('RGBA')
                    
                    # Save as PNG to preserve transparency
                    buffered = BytesIO()
                    img.save(buffered, format="PNG", optimize=True)
                    result_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    
                    logger.info("Background removal successful with transparency")
                    return result_base64
                else:
                    logger.error(f"Failed to download result: {img_response.status_code}")
                    return None
                    
            elif status == 'failed':
                error = result.get('error', 'Unknown error')
                logger.error(f"Prediction failed: {error}")
                return None
            
            # Still processing
            if attempt == max_attempts - 1:
                logger.info("Timeout waiting for prediction")
                return None
        
        return None
        
    except Exception as e:
        logger.error(f"Replicate API error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def process_thumbnail(job):
    """Process thumbnail request"""
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
        color = 'yellow_gold'  # default
        replicate_token = None
        
        if isinstance(input_data, dict):
            # Extract image
            for key in ['image', 'image_base64', 'imageBase64', 'base64_image', 
                       'url', 'image_url', 'imageUrl', 'enhanced_image']:
                if key in input_data:
                    image_data = input_data[key]
                    break
            
            # Extract color
            color = input_data.get('color', 'yellow_gold')
            
            # Extract Replicate token
            replicate_token = input_data.get('replicate_api_token', '')
            
        elif isinstance(input_data, str):
            image_data = input_data
        
        # Get Replicate token from job_input if not found
        if not replicate_token and isinstance(job_input, dict):
            replicate_token = job_input.get('replicate_api_token', '')
        
        # V64: Use environment variable token if no token provided
        if not replicate_token:
            replicate_token = REPLICATE_API_TOKEN
            logger.info("Using Replicate API token from environment variable")
        
        if not image_data:
            return {
                "output": {
                    "error": "Could not extract image data from input",
                    "status": "error"
                }
            }
        
        if not replicate_token:
            return {
                "output": {
                    "error": "Replicate API token is required (not found in input or environment)",
                    "status": "error"
                }
            }
        
        # Color mapping
        color_map = {
            'yellow_gold': '#FFD700',
            'rose_gold': '#E8B4B8',
            'white_gold': '#F5F5F5',
            'white': '#FFFFFF'
        }
        
        # Process based on data type
        if isinstance(image_data, str) and image_data.startswith('http'):
            # URL input
            logger.info(f"Processing URL: {image_data[:100]}...")
            image = download_image_from_url(image_data)
        else:
            # Base64 input
            logger.info("Processing base64 image...")
            
            # Validate base64
            is_valid, clean_base64 = validate_base64(image_data)
            if not is_valid:
                return {
                    "output": {
                        "error": "Invalid base64 image data",
                        "status": "error"
                    }
                }
            
            # Decode image
            image_bytes = decode_base64_safe(clean_base64)
            image = Image.open(BytesIO(image_bytes))
        
        logger.info(f"Original image: {image.mode} {image.size}")
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            logger.info(f"Converting from {image.mode} to RGB")
            image = image.convert('RGB')
        
        # Step 1: Center crop to square
        logger.info("Cropping to square...")
        image = center_crop_to_square(image)
        
        # Step 2: Resize to 800x800 for Replicate
        logger.info("Resizing to 800x800...")
        image = image.resize((800, 800), Image.Resampling.LANCZOS)
        
        # Convert to base64 for Replicate
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Step 3: Remove background using Replicate
        logger.info("Removing background with Replicate...")
        result_base64 = remove_background_with_replicate(image_base64, replicate_token)
        
        if not result_base64:
            # Fallback: return original image with error message
            logger.warning("Background removal failed, returning original")
            result_base64 = image_base64
            status_message = "Background removal failed - returning original"
        else:
            status_message = "Thumbnail created with transparent background - V64"
        
        # Step 4: Apply final processing
        # Decode the result
        result_bytes = decode_base64_safe(result_base64)
        result_image = Image.open(BytesIO(result_bytes))
        
        # Ensure RGBA mode for transparency
        if result_image.mode != 'RGBA':
            result_image = result_image.convert('RGBA')
        
        # Step 5: Create 1000x1300 thumbnail
        logger.info("Creating final thumbnail...")
        thumbnail = create_thumbnail_with_crop(result_image, (1000, 1300))
        
        # Step 6: Detect color from the thumbnail
        detected_color = detect_ring_color(thumbnail)
        logger.info(f"Detected color: {detected_color}")
        
        # Step 7: Apply color-specific enhancements
        thumbnail = apply_color_specific_enhancement(thumbnail, detected_color)
        
        # Final sharpening
        thumbnail = thumbnail.filter(ImageFilter.UnsharpMask(radius=1.0, percent=100, threshold=3))
        
        # Convert to base64
        buffered = BytesIO()
        thumbnail.save(buffered, format="PNG", optimize=True)
        thumbnail_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Prepare results with different padding options
        # For Make.com - no padding
        result_base64_no_padding = thumbnail_base64.rstrip('=')
        
        # For Google Script - with padding
        result_base64_with_padding = thumbnail_base64
        padding_needed = 4 - (len(thumbnail_base64) % 4)
        if padding_needed and padding_needed != 4:
            result_base64_with_padding = thumbnail_base64 + ('=' * padding_needed)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        logger.info(f"Thumbnail processing completed in {processing_time:.2f}s")
        logger.info(f"Output size: {len(result_base64_no_padding)} chars (no padding)")
        logger.info(f"Output size: {len(result_base64_with_padding)} chars (with padding)")
        
        return {
            "output": {
                "thumbnail": result_base64_no_padding,  # Make.com용 (padding 없음)
                "thumbnail_with_padding": result_base64_with_padding,  # Google Script용
                "color": color_map.get(detected_color, '#FFD700'),
                "status": "success",
                "message": status_message,
                "processing_time": f"{processing_time:.2f}s",
                "detected_color": detected_color,
                "original_size": list(image.size),
                "final_size": list(thumbnail.size),
                "settings": {
                    "size": "1000x1300",
                    "background": "transparent",
                    "color_name": detected_color,
                    "color_hex": color_map.get(detected_color, '#FFD700'),
                    "sharpness": "150%"
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
