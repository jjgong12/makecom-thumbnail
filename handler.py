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

VERSION = "V5-FixedCenterCrop-EnhancedQuality"

def find_input_data(data):
    """Find input data recursively"""
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

def find_filename_enhanced(data, depth=0):
    """Enhanced filename extraction"""
    if depth > 10:
        return None
    
    found_filenames = []
    
    def extract_filenames(obj, current_depth):
        if current_depth > 10:
            return
            
        if isinstance(obj, dict):
            filename_keys = [
                'filename', 'file_name', 'fileName', 'name', 'file',
                'originalName', 'original_name', 'originalFileName', 'original_file_name',
                'image_name', 'imageName', 'imageFileName', 'image_file_name',
                'ring_filename', 'ringFilename', 'product_name', 'productName',
                'title', 'label', 'id', 'identifier', 'reference'
            ]
            
            for key, value in obj.items():
                if isinstance(value, str) and any(pattern in value.lower() for pattern in ['ac_', 'bc_', 'a_', 'b_', 'c_']):
                    if len(value) < 100:
                        found_filenames.append(value)
                        logger.info(f"Found potential filename in value: {value}")
                
                if key.lower() in [k.lower() for k in filename_keys]:
                    if isinstance(value, str) and value and len(value) < 100:
                        found_filenames.append(value)
                        logger.info(f"Found filename at key '{key}': {value}")
                
                if isinstance(value, dict):
                    extract_filenames(value, current_depth + 1)
                elif isinstance(value, list):
                    for item in value:
                        extract_filenames(item, current_depth + 1)
    
    extract_filenames(data, 0)
    
    if found_filenames:
        filename = found_filenames[0]
        logger.info(f"Selected filename: {filename}")
        return filename
    
    return None

def generate_thumbnail_filename(original_filename, image_index):
    """Generate thumbnail filename with 007, 008, 009 pattern"""
    if not original_filename:
        return f"thumbnail_{image_index:03d}.jpg"
    
    import re
    
    # No conversion needed - already b_ or bc_ pattern
    new_filename = original_filename
    
    # Map image index to 007, 008, 009
    # 원본 6장 중 3장만 선택 (예: 1,3,5번째 → 007,008,009)
    thumbnail_numbers = {
        1: "007",  # First selected image (from 001 or 002)
        3: "008",  # Second selected image (from 003 or 004)
        5: "009"   # Third selected image (from 005 or 006)
    }
    
    # Replace the number part with new number
    pattern = r'(_\d{3})'
    if re.search(pattern, new_filename):
        new_filename = re.sub(pattern, f'_{thumbnail_numbers.get(image_index, "007")}', new_filename)
    else:
        # If no number pattern found, append it
        name_parts = new_filename.split('.')
        name_parts[0] += f'_{thumbnail_numbers.get(image_index, "007")}'
        new_filename = '.'.join(name_parts)
    
    logger.info(f"Generated thumbnail filename: {original_filename} -> {new_filename}")
    return new_filename

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

def detect_pattern_type(filename: str) -> str:
    """Detect pattern type"""
    if not filename:
        return "other"
    
    filename_lower = filename.lower()
    logger.info(f"Checking filename pattern: {filename_lower}")
    
    import re
    
    pattern_ac_bc = re.search(r'(ac_|bc_)', filename_lower)
    pattern_a_only = re.search(r'(?<!a)(?<!b)a_', filename_lower)
    
    if pattern_ac_bc:
        logger.info("Pattern detected: ac_ or bc_ (unplated white)")
        return "ac_bc"
    elif pattern_a_only:
        logger.info("Pattern detected: a_ only")
        return "a_only"
    else:
        logger.info("Pattern detected: other")
        return "other"

def apply_basic_enhancement(image):
    """Apply basic enhancement - same as enhancement handler V121"""
    if image.mode != 'RGB':
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        else:
            image = image.convert('RGB')
    
    # V121 settings - enhanced brightness
    brightness = ImageEnhance.Brightness(image)
    image = brightness.enhance(1.02)  # Enhanced from V120
    
    contrast = ImageEnhance.Contrast(image)
    image = contrast.enhance(1.02)  # Enhanced from V120
    
    color = ImageEnhance.Color(image)
    image = color.enhance(1.01)  # Enhanced from V120
    
    return image

def apply_color_specific_enhancement(image, pattern_type, filename):
    """Apply enhancement based on pattern type - V5 with V121 brightness"""
    
    logger.info(f"Applying enhancement - Filename: {filename}, Pattern type: {pattern_type}")
    
    if pattern_type == "ac_bc":
        logger.info("Applying unplated white enhancement (15% white overlay)")
        
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.02)  # Enhanced to match V121
        
        color = ImageEnhance.Color(image)
        image = color.enhance(0.96)
        
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.0)
        
        # V5: 15% white overlay (same as V121)
        img_array = np.array(image)
        img_array = img_array * 0.85 + 255 * 0.15
        image = Image.fromarray(img_array.astype(np.uint8))
        
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.00)
        
    elif pattern_type == "a_only":
        logger.info("Applying a_ pattern enhancement")
        
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.03)  # Enhanced to match V121
        
        color = ImageEnhance.Color(image)
        image = color.enhance(0.98)
        
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.01)  # Enhanced to match V121
        
        width, height = image.size
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        X, Y = np.meshgrid(x, y)
        distance = np.sqrt(X**2 + Y**2)
        
        focus_mask = 1 + 0.025 * np.exp(-distance**2 * 1.0)  # Enhanced to match V121
        focus_mask = np.clip(focus_mask, 1.0, 1.025)
        
        img_array = np.array(image)
        for i in range(3):
            img_array[:, :, i] = np.clip(img_array[:, :, i] * focus_mask, 0, 255)
        image = Image.fromarray(img_array.astype(np.uint8))
        
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.00)
        
    else:
        logger.info("Standard enhancement")
        
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.02)  # Enhanced to match V121
        
        color = ImageEnhance.Color(image)
        image = color.enhance(0.98)
        
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.01)  # Enhanced to match V121
        
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.00)
    
    return image

def apply_spotlight_effect(image):
    """Apply subtle spotlight effect to center of image"""
    logger.info("Applying spotlight effect")
    
    width, height = image.size
    
    # Create radial gradient for spotlight
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)
    
    # Distance from center
    distance = np.sqrt(X**2 + Y**2)
    
    # Create spotlight mask - enhanced for V5
    spotlight_mask = 1 + 0.025 * np.exp(-distance**2 * 1.2)  # Enhanced from 0.02 to 0.025
    spotlight_mask = np.clip(spotlight_mask, 1.0, 1.025)
    
    # Apply spotlight
    img_array = np.array(image)
    for i in range(3):
        img_array[:, :, i] = np.clip(img_array[:, :, i] * spotlight_mask, 0, 255)
    
    return Image.fromarray(img_array.astype(np.uint8))

def create_thumbnail_smart_center_crop_with_upscale(image, target_width=1000, target_height=1300):
    """Create thumbnail with FIXED CENTER CROP and upscaling - V5 always uses image center"""
    original_width, original_height = image.size
    
    # V5: ALWAYS use image center - DO NOT detect ring center
    image_center = (original_width // 2, original_height // 2)
    logger.info(f"Using FIXED image center: {image_center}")
    
    # V5: Apply upscaling first if image is smaller than target
    if original_width < target_width or original_height < target_height:
        logger.info(f"Image {original_width}x{original_height} smaller than target, upscaling first")
        
        # Calculate scale factor to ensure both dimensions are at least target size
        scale_factor = max(target_width / original_width, target_height / original_height) * 1.1  # 10% extra
        
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        
        # Upscale using LANCZOS for best quality
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        logger.info(f"Upscaled to {new_width}x{new_height}")
        
        # Adjust center coordinates proportionally
        image_center = (int(image_center[0] * scale_factor), int(image_center[1] * scale_factor))
        original_width, original_height = new_width, new_height
    
    # Check for specific size ranges - V5 includes 3000x3900 support
    if ((1800 <= original_width <= 2200 and 2400 <= original_height <= 2800) or
        (2800 <= original_width <= 3200 and 3700 <= original_height <= 4100)):
        
        if original_width >= 2800:
            logger.info(f"Input {original_width}x{original_height} detected as ~3000x3900, using fixed center crop")
            crop_ratio = 0.75  # Smaller crop ratio for larger images
        else:
            logger.info(f"Input {original_width}x{original_height} detected as ~2000x2600, using fixed center crop")
            crop_ratio = 0.85
        
        crop_width = int(original_width * crop_ratio)
        crop_height = int(original_height * crop_ratio)
        
        # V5: FIXED center crop - always use image center
        left = max(0, image_center[0] - crop_width // 2)
        top = max(0, image_center[1] - crop_height // 2)
        
        # Adjust if crop goes beyond image bounds
        if left + crop_width > original_width:
            left = original_width - crop_width
        if top + crop_height > original_height:
            top = original_height - crop_height
        
        right = left + crop_width
        bottom = top + crop_height
        
        logger.info(f"Fixed center crop coordinates: ({left}, {top}, {right}, {bottom})")
        
        cropped = image.crop((left, top, right, bottom))
        
        return cropped.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
    else:
        logger.info(f"Input {original_width}x{original_height} not in specific ranges, using aspect ratio preservation")
        
        width_ratio = target_width / original_width
        height_ratio = target_height / original_height
        
        scale_ratio = min(width_ratio, height_ratio)
        
        new_width = int(original_width * scale_ratio)
        new_height = int(original_height * scale_ratio)
        
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        thumbnail = Image.new('RGB', (target_width, target_height), (255, 255, 255))
        
        left = (target_width - new_width) // 2
        top = (target_height - new_height) // 2
        
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
    
    img_base64_no_padding = img_base64.rstrip('=')
    
    return img_base64_no_padding

def handler(event):
    """Thumbnail handler function - V5 with fixed center crop and enhanced quality"""
    try:
        logger.info(f"Thumbnail {VERSION} started")
        logger.info(f"Input data type: {type(event)}")
        
        # Get image index (1, 3, or 5)
        image_index = event.get('image_index', 1)
        if isinstance(event.get('input'), dict):
            image_index = event.get('input', {}).get('image_index', image_index)
        
        # Use enhanced filename detection
        filename = find_filename_enhanced(event)
        if filename:
            logger.info(f"Successfully extracted filename: {filename}")
        else:
            logger.warning("Could not extract filename from input - will use default enhancement")
        
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
        
        # 1. Apply basic enhancement (same as V121)
        enhanced_image = apply_basic_enhancement(image)
        
        # 2. Smart thumbnail creation with FIXED CENTER CROP and UPSCALING - V5 with fixed center
        thumbnail = create_thumbnail_smart_center_crop_with_upscale(enhanced_image, 1000, 1300)
        
        # 3. Detect pattern type
        pattern_type = detect_pattern_type(filename)
        
        if pattern_type == "ac_bc":
            detected_type = "무도금화이트"
        elif pattern_type == "a_only":
            detected_type = "a_패턴"
        else:
            detected_type = "기타색상"
        
        logger.info(f"Final detection - Type: {detected_type}, Filename: {filename}")
        
        # 4. Apply pattern-specific enhancement (with V121 brightness)
        thumbnail = apply_color_specific_enhancement(thumbnail, pattern_type, filename)
        
        # 5. Apply enhanced spotlight effect
        thumbnail = apply_spotlight_effect(thumbnail)
        
        # 6. Enhanced sharpness for details
        sharpness = ImageEnhance.Sharpness(thumbnail)
        thumbnail = sharpness.enhance(1.25)  # Enhanced from 1.20
        
        # 7. Final brightness touch - enhanced
        brightness = ImageEnhance.Brightness(thumbnail)
        thumbnail = brightness.enhance(1.02)  # Enhanced from 1.01
        
        # Convert to base64
        thumbnail_base64 = image_to_base64(thumbnail)
        
        # Generate output filename (007, 008, or 009)
        output_filename = generate_thumbnail_filename(filename, image_index)
        
        # Return result
        return {
            "output": {
                "thumbnail": thumbnail_base64,
                "size": list(thumbnail.size),
                "detected_type": detected_type,
                "pattern_type": pattern_type,
                "filename": output_filename,  # Will be b_007, bc_007, etc.
                "original_filename": filename,
                "image_index": image_index,
                "format": "base64_no_padding",
                "has_spotlight": True,
                "has_upscaling": True,
                "supports_3000x3900": True,  # V5 feature
                "fixed_center_crop": True,  # V5 NEW feature
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
