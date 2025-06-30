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

VERSION = "V2-ThumbnailFor789"

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
    
    # Extract pattern prefix (a_, ac_, etc.)
    import re
    
    # Convert a_ to b_, ac_ to bc_
    new_filename = original_filename
    if original_filename.startswith('a_'):
        new_filename = 'b_' + original_filename[2:]
    elif original_filename.startswith('ac_'):
        new_filename = 'bc_' + original_filename[3:]
    
    # Map image index to 007, 008, 009
    # 원본 6장 중 3장만 선택 (예: 1,3,5번째 → 007,008,009)
    thumbnail_numbers = {
        1: "007",  # First selected image
        3: "008",  # Second selected image  
        5: "009"   # Third selected image
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
    """Apply basic enhancement"""
    if image.mode != 'RGB':
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        else:
            image = image.convert('RGB')
    
    brightness = ImageEnhance.Brightness(image)
    image = brightness.enhance(1.08)
    
    contrast = ImageEnhance.Contrast(image)
    image = contrast.enhance(1.05)
    
    color = ImageEnhance.Color(image)
    image = color.enhance(1.03)
    
    return image

def apply_color_specific_enhancement(image, pattern_type, filename):
    """Apply enhancement based on pattern type"""
    
    logger.info(f"Applying enhancement - Filename: {filename}, Pattern type: {pattern_type}")
    
    if pattern_type == "ac_bc":
        logger.info("Applying unplated white enhancement (28% white overlay)")
        
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.04)
        
        color = ImageEnhance.Color(image)
        image = color.enhance(0.92)
        
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.0)
        
        img_array = np.array(image)
        img_array = img_array * 0.72 + 255 * 0.28
        image = Image.fromarray(img_array.astype(np.uint8))
        
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.00)
        
    elif pattern_type == "a_only":
        logger.info("Applying a_ pattern enhancement")
        
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.08)
        
        color = ImageEnhance.Color(image)
        image = color.enhance(0.94)
        
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.02)
        
        width, height = image.size
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        X, Y = np.meshgrid(x, y)
        distance = np.sqrt(X**2 + Y**2)
        
        focus_mask = 1 + 0.06 * np.exp(-distance**2 * 0.7)
        focus_mask = np.clip(focus_mask, 1.0, 1.06)
        
        img_array = np.array(image)
        for i in range(3):
            img_array[:, :, i] = np.clip(img_array[:, :, i] * focus_mask, 0, 255)
        image = Image.fromarray(img_array.astype(np.uint8))
        
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.02)
        
    else:
        logger.info("Standard enhancement")
        
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.04)
        
        color = ImageEnhance.Color(image)
        image = color.enhance(0.92)
        
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.0)
        
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.00)
    
    return image

def create_thumbnail_smart_center_crop(image, target_width=1000, target_height=1300):
    """Create thumbnail with center crop for ~2000x2600 input to fill 90% of frame"""
    original_width, original_height = image.size
    
    if (1800 <= original_width <= 2200 and 2400 <= original_height <= 2800):
        logger.info(f"Input {original_width}x{original_height} detected as ~2000x2600, using center crop")
        
        crop_ratio = 0.85
        
        crop_width = int(original_width * crop_ratio)
        crop_height = int(original_height * crop_ratio)
        
        left = (original_width - crop_width) // 2
        top = (original_height - crop_height) // 2
        right = left + crop_width
        bottom = top + crop_height
        
        cropped = image.crop((left, top, right, bottom))
        
        return cropped.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
    else:
        logger.info(f"Input {original_width}x{original_height} not ~2000x2600, using aspect ratio preservation")
        
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
    """Thumbnail handler function - creates thumbnails 007, 008, 009 from selected images"""
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
        
        # 1. Apply basic enhancement
        enhanced_image = apply_basic_enhancement(image)
        
        # 2. Smart thumbnail creation with CENTER CROP
        thumbnail = create_thumbnail_smart_center_crop(enhanced_image, 1000, 1300)
        
        # 3. Detect pattern type
        pattern_type = detect_pattern_type(filename)
        
        if pattern_type == "ac_bc":
            detected_type = "무도금화이트"
        elif pattern_type == "a_only":
            detected_type = "a_패턴"
        else:
            detected_type = "기타색상"
        
        logger.info(f"Final detection - Type: {detected_type}, Filename: {filename}")
        
        # 4. Apply pattern-specific enhancement
        thumbnail = apply_color_specific_enhancement(thumbnail, pattern_type, filename)
        
        # 5. Light sharpness for details
        sharpness = ImageEnhance.Sharpness(thumbnail)
        thumbnail = sharpness.enhance(1.1)
        
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
