import runpod
import os
import json
import time
import base64
import numpy as np
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VERSION = "V95-1.8PercentWhiteOverlay"

def find_input_data(data):
    """Find input data recursively - matches Enhancement handler"""
    if isinstance(data, dict):
        if any(key in data for key in ['image', 'url', 'image_url', 'imageUrl', 'image_base64', 'imageBase64']):
            return data
        
        if 'input' in data:
            return find_input_data(data['input'])
        
        for key in ['job', 'payload', 'data']:
            if key in data:
                result = find_input_data(data[key])
                if result:
                    return result
    return data

def find_filename(data, depth=0):
    """Extract filename from input data - IMPROVED for Make.com"""
    if depth > 5:  # Prevent infinite recursion
        return None
        
    if isinstance(data, dict):
        # Check common filename keys - EXPANDED LIST
        filename_keys = ['filename', 'file_name', 'name', 'fileName', 'file', 
                        'originalName', 'original_name', 'image_name', 'imageName']
        
        # Log the keys at current level for debugging
        if depth == 0:
            logger.info(f"Top level keys: {list(data.keys())[:20]}")  # First 20 keys
        
        for key in filename_keys:
            if key in data and isinstance(data[key], str):
                logger.info(f"Found filename at key '{key}': {data[key]}")
                return data[key]
        
        # Deep recursive search through ALL keys
        for key, value in data.items():
            if isinstance(value, dict):
                result = find_filename(value, depth + 1)
                if result:
                    return result
            elif isinstance(value, list) and len(value) > 0:
                for item in value:
                    if isinstance(item, dict):
                        result = find_filename(item, depth + 1)
                        if result:
                            return result
    
    return None

def download_image_from_url(url):
    """Download image from URL"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8'
    }
    
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))

def base64_to_image(base64_string):
    """Convert base64 to PIL Image"""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    padding = 4 - len(base64_string) % 4
    if padding != 4:
        base64_string += '=' * padding
    
    img_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(img_data))

def detect_if_unplated_white(filename: str) -> bool:
    """Check if filename indicates unplated white (contains 'c')"""
    if not filename:
        logger.warning("No filename found, defaulting to standard enhancement")
        return False
    
    # Convert to lowercase for case-insensitive check
    filename_lower = filename.lower()
    logger.info(f"Checking filename pattern: {filename_lower}")
    
    # Check for ac_, bc_, dc_, etc. patterns
    import re
    # Pattern: any letter followed by c_ or just c_
    pattern1 = re.search(r'[a-z]?c_', filename_lower)
    # Pattern: any letter followed by c. or just c.
    pattern2 = re.search(r'[a-z]?c\.', filename_lower)
    # Pattern: c followed by number
    pattern3 = re.match(r'^c\d', filename_lower)
    
    is_unplated = bool(pattern1 or pattern2 or pattern3)
    
    logger.info(f"Pattern check - ac_/bc_/c_: {bool(pattern1)}, c.: {bool(pattern2)}, c+digit: {bool(pattern3)}")
    logger.info(f"Is unplated white: {is_unplated}")
    
    return is_unplated

def apply_basic_enhancement(image):
    """Apply basic enhancement matching Enhancement V95"""
    if image.mode != 'RGB':
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        else:
            image = image.convert('RGB')
    
    # Match Enhancement V95 basic settings
    brightness = ImageEnhance.Brightness(image)
    image = brightness.enhance(1.08)
    
    contrast = ImageEnhance.Contrast(image)
    image = contrast.enhance(1.05)
    
    color = ImageEnhance.Color(image)
    image = color.enhance(1.03)
    
    return image

def apply_color_specific_enhancement(image, is_unplated_white, filename):
    """Apply enhancement - 1.8% WHITE OVERLAY ONLY FOR UNPLATED WHITE (filename with 'c')"""
    
    logger.info(f"Applying enhancement - Filename: {filename}, Is unplated white: {is_unplated_white}")
    
    if is_unplated_white:
        # UPDATED TO 1.8% WHITE EFFECT FOR V95!
        logger.info("Applying unplated white enhancement (1.8% white overlay)")
        
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.08)
        
        color = ImageEnhance.Color(image)
        image = color.enhance(0.5)  # Keep more color
        
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.0)  # No contrast change
        
        # UPDATED: 1.8% white mixing (was 1.5%)
        img_array = np.array(image)
        img_array = img_array * 0.982 + 255 * 0.018  # 1.8% white overlay
        image = Image.fromarray(img_array.astype(np.uint8))
        
        # Very tiny additional boost
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.01)  # Minimal boost
        
    else:
        # For all other colors - NO white overlay
        logger.info("Standard enhancement (no white overlay)")
        
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.06)
        
        color = ImageEnhance.Color(image)
        image = color.enhance(1.05)
    
    return image

def apply_lighting_effect(image):
    """Apply subtle spotlight/lighting effect"""
    width, height = image.size
    
    # Create radial gradient for spotlight
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)
    
    # Offset spotlight slightly up and left for natural lighting
    X_offset = X + 0.2
    Y_offset = Y + 0.3
    
    # Distance from offset center
    distance = np.sqrt(X_offset**2 + Y_offset**2)
    
    # Create spotlight effect (brighter in center)
    spotlight = 1 + 0.08 * np.exp(-distance**2 * 1.5)
    spotlight = np.clip(spotlight, 1.0, 1.08)
    
    # Apply spotlight
    img_array = np.array(image)
    for i in range(3):
        img_array[:, :, i] = np.clip(img_array[:, :, i] * spotlight, 0, 255)
    
    # Add subtle highlight on upper area
    highlight_mask = np.exp(-Y * 3) * 0.02
    highlight_mask = np.clip(highlight_mask, 0, 0.02)
    
    for i in range(3):
        img_array[:, :, i] = np.clip(img_array[:, :, i] + highlight_mask * 255, 0, 255)
    
    return Image.fromarray(img_array.astype(np.uint8))

def create_thumbnail_smart(image, target_width=1000, target_height=1300):
    """Create thumbnail with smart handling for ~2000x2600 input (±200 pixels)"""
    original_width, original_height = image.size
    
    # Check if input is approximately 2000x2600 (±200 pixels)
    if (1800 <= original_width <= 2200 and 2400 <= original_height <= 2800):
        # Force resize to exact 1000x1300 without padding
        logger.info(f"Input {original_width}x{original_height} detected as ~2000x2600, forcing exact resize")
        return image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    else:
        # For other sizes, maintain aspect ratio with padding
        logger.info(f"Input {original_width}x{original_height} not ~2000x2600, using aspect ratio preservation")
        
        # Calculate scaling to fit within target dimensions
        width_ratio = target_width / original_width
        height_ratio = target_height / original_height
        
        # Use the smaller ratio to ensure the image fits within bounds
        scale_ratio = min(width_ratio, height_ratio)
        
        # Calculate new dimensions
        new_width = int(original_width * scale_ratio)
        new_height = int(original_height * scale_ratio)
        
        # Resize image maintaining aspect ratio
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create white background
        thumbnail = Image.new('RGB', (target_width, target_height), (255, 255, 255))
        
        # Calculate position to center the image
        left = (target_width - new_width) // 2
        top = (target_height - new_height) // 2
        
        # Paste resized image onto white background
        thumbnail.paste(resized, (left, top))
        
        return thumbnail

def image_to_base64(image):
    """Convert image to base64 without padding for Make.com"""
    buffered = BytesIO()
    
    if image.mode == 'RGBA':
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
    
    image.save(buffered, format='PNG', quality=95)
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    # Remove padding for Make.com
    img_base64_no_padding = img_base64.rstrip('=')
    
    return img_base64_no_padding

def handler(event):
    """Thumbnail handler function"""
    try:
        logger.info(f"Thumbnail {VERSION} started")
        logger.info(f"Input data type: {type(event)}")
        
        # Find filename FIRST - IMPROVED
        filename = find_filename(event)
        if filename:
            logger.info(f"Successfully extracted filename: {filename}")
        else:
            logger.warning("Could not extract filename from input")
            # Log the structure for debugging
            if isinstance(event, dict):
                logger.info(f"Event structure keys: {list(event.keys())[:10]}")
        
        # Find input data
        input_data = find_input_data(event)
        
        if not input_data:
            raise ValueError("No input data found")
        
        # Get image
        image = None
        
        if isinstance(input_data, dict):
            if 'image_base64' in input_data or 'imageBase64' in input_data:
                base64_str = input_data.get('image_base64') or input_data.get('imageBase64')
                image = base64_to_image(base64_str)
            elif 'url' in input_data or 'image_url' in input_data or 'imageUrl' in input_data:
                image_url = input_data.get('url') or input_data.get('image_url') or input_data.get('imageUrl')
                image = download_image_from_url(image_url)
        elif isinstance(input_data, str):
            if input_data.startswith('http'):
                image = download_image_from_url(input_data)
            else:
                image = base64_to_image(input_data)
        
        if not image:
            raise ValueError("Failed to load image")
        
        logger.info(f"Image loaded: {image.size}")
        
        # 1. Apply basic enhancement (matching Enhancement V95)
        enhanced_image = apply_basic_enhancement(image)
        
        # 2. Smart thumbnail creation (no padding for ~2000x2600 ±200px)
        thumbnail = create_thumbnail_smart(enhanced_image, 1000, 1300)
        
        # 3. Check if unplated white based on filename
        is_unplated_white = detect_if_unplated_white(filename)
        detected_type = "무도금화이트" if is_unplated_white else "기타색상"
        logger.info(f"Final detection - Type: {detected_type}, Filename: {filename}")
        
        # 4. Apply color-specific enhancement (1.8% white overlay only for 'c' filenames)
        thumbnail = apply_color_specific_enhancement(thumbnail, is_unplated_white, filename)
        
        # 5. Apply lighting effect
        thumbnail = apply_lighting_effect(thumbnail)
        
        # 6. Light sharpness for details
        sharpness = ImageEnhance.Sharpness(thumbnail)
        thumbnail = sharpness.enhance(1.1)
        
        # Convert to base64
        thumbnail_base64 = image_to_base64(thumbnail)
        
        # Return result
        return {
            "output": {
                "thumbnail": thumbnail_base64,
                "size": list(thumbnail.size),
                "detected_type": detected_type,
                "filename": filename,
                "format": "base64_no_padding",
                "version": VERSION,
                "status": "success"
            }
        }
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        return {
            "output": {
                "error": str(e),
                "status": "failed",
                "version": VERSION,
                "traceback": traceback.format_exc()
            }
        }

# RunPod handler
runpod.serverless.start({"handler": handler})
