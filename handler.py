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

VERSION = "V111-5PercentWhiteOverlay-UnifiedBrightness"

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

def find_filename_enhanced(data, depth=0):
    """Enhanced filename extraction for Make.com - checks EVERYWHERE"""
    if depth > 10:  # Increased depth limit
        return None
    
    found_filenames = []
    
    def extract_filenames(obj, current_depth):
        """Recursively extract all potential filenames"""
        if current_depth > 10:
            return
            
        if isinstance(obj, dict):
            # Extended list of filename keys
            filename_keys = [
                'filename', 'file_name', 'fileName', 'name', 'file',
                'originalName', 'original_name', 'originalFileName', 'original_file_name',
                'image_name', 'imageName', 'imageFileName', 'image_file_name',
                'ring_filename', 'ringFilename', 'product_name', 'productName',
                'title', 'label', 'id', 'identifier', 'reference'
            ]
            
            # Check all keys
            for key, value in obj.items():
                # Check if key itself looks like a filename pattern
                if isinstance(value, str) and any(pattern in value.lower() for pattern in ['ac_', 'bc_', 'a_', 'b_', 'c_']):
                    if len(value) < 100:  # Reasonable filename length
                        found_filenames.append(value)
                        logger.info(f"Found potential filename in value: {value}")
                
                # Check known filename keys
                if key.lower() in [k.lower() for k in filename_keys]:
                    if isinstance(value, str) and value and len(value) < 100:
                        found_filenames.append(value)
                        logger.info(f"Found filename at key '{key}': {value}")
                
                # Recursive search
                if isinstance(value, dict):
                    extract_filenames(value, current_depth + 1)
                elif isinstance(value, list):
                    for item in value:
                        extract_filenames(item, current_depth + 1)
                elif isinstance(value, str):
                    # Check if the string itself might contain JSON
                    if value.startswith('{') and value.endswith('}'):
                        try:
                            parsed = json.loads(value)
                            extract_filenames(parsed, current_depth + 1)
                        except:
                            pass
        
        elif isinstance(obj, list):
            for item in obj:
                extract_filenames(item, current_depth)
    
    # Start extraction
    extract_filenames(data, 0)
    
    # Filter and prioritize filenames
    valid_filenames = []
    for fname in found_filenames:
        # Check if it looks like our ring filename pattern
        if any(pattern in fname.lower() for pattern in ['ac_', 'bc_', 'a_', 'b_', 'c_']):
            valid_filenames.append(fname)
    
    if valid_filenames:
        # Return the most likely filename (first one with our pattern)
        filename = valid_filenames[0]
        logger.info(f"Selected filename: {filename} from {len(valid_filenames)} candidates")
        return filename
    
    # Log all keys at root level for debugging
    if depth == 0 and isinstance(data, dict):
        logger.info(f"Root level keys: {list(data.keys())}")
        # Also log some values to see structure
        for key in list(data.keys())[:5]:
            logger.info(f"Key '{key}' type: {type(data[key])}")
    
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
    """Check if filename indicates unplated white (ONLY ac_ or bc_ patterns)"""
    if not filename:
        logger.warning("No filename found, defaulting to standard enhancement")
        return False
    
    # Convert to lowercase for case-insensitive check
    filename_lower = filename.lower()
    logger.info(f"Checking filename pattern: {filename_lower}")
    
    # Check ONLY for ac_ or bc_ patterns (NOT just c_)
    import re
    # Pattern: specifically ac_ or bc_
    pattern_ac_bc = re.search(r'(ac_|bc_)', filename_lower)
    
    is_unplated = bool(pattern_ac_bc)
    
    logger.info(f"Pattern check - ac_/bc_: {bool(pattern_ac_bc)}")
    logger.info(f"Is unplated white: {is_unplated}")
    
    return is_unplated

def apply_basic_enhancement(image):
    """Apply basic enhancement matching Enhancement V111 - brighter"""
    if image.mode != 'RGB':
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        else:
            image = image.convert('RGB')
    
    # Match Enhancement V111 basic settings - brighter
    brightness = ImageEnhance.Brightness(image)
    image = brightness.enhance(1.10)  # Increased from 1.09
    
    contrast = ImageEnhance.Contrast(image)
    image = contrast.enhance(1.05)
    
    color = ImageEnhance.Color(image)
    image = color.enhance(1.03)
    
    return image

def apply_color_specific_enhancement(image, is_unplated_white, filename):
    """Apply enhancement - 5% WHITE OVERLAY with unified brightness settings"""
    
    logger.info(f"Applying enhancement - Filename: {filename}, Is unplated white: {is_unplated_white}")
    
    # V111: Unified brightness settings for ALL colors
    # First brightness adjustment (same for all)
    brightness = ImageEnhance.Brightness(image)
    image = brightness.enhance(1.14)  # Same for all colors
    
    # Color adjustment (same for all)
    color = ImageEnhance.Color(image)
    image = color.enhance(0.92)  # Same for all colors
    
    # Contrast (same for all)
    contrast = ImageEnhance.Contrast(image)
    image = contrast.enhance(1.0)  # Same for all colors
    
    # Apply white overlay ONLY for unplated white
    if is_unplated_white:
        # V111: 5% white overlay
        logger.info("Applying unplated white enhancement (5% white overlay)")
        img_array = np.array(image)
        img_array = img_array * 0.95 + 255 * 0.05  # 5% white overlay
        image = Image.fromarray(img_array.astype(np.uint8))
    else:
        logger.info("Standard enhancement (no white overlay)")
    
    # Final brightness boost (same for all)
    brightness = ImageEnhance.Brightness(image)
    image = brightness.enhance(1.05)  # Same for all colors
    
    return image

def apply_background_whitening(image):
    """Apply background whitening effect"""
    img_array = np.array(image)
    
    # Create a subtle vignette that brightens the edges
    height, width = img_array.shape[:2]
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)
    
    # Distance from center
    distance = np.sqrt(X**2 + Y**2)
    
    # Invert for edge brightening (brighter at edges)
    edge_mask = np.clip(distance * 0.5, 0, 1)  # 0 at center, 1 at edges
    
    # Apply white overlay to edges
    for i in range(3):
        img_array[:, :, i] = img_array[:, :, i] * (1 - edge_mask * 0.08) + 255 * edge_mask * 0.08
    
    return Image.fromarray(img_array.astype(np.uint8))

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

def create_thumbnail_smart_center_crop(image, target_width=1000, target_height=1300):
    """Create thumbnail with center crop for ~2000x2600 input to fill 90% of frame"""
    original_width, original_height = image.size
    
    # Check if input is approximately 2000x2600 (±200 pixels)
    if (1800 <= original_width <= 2200 and 2400 <= original_height <= 2800):
        logger.info(f"Input {original_width}x{original_height} detected as ~2000x2600, using center crop")
        
        # Calculate crop to make rings fill 90% of the frame
        # We want to crop the center part and then resize
        crop_ratio = 0.85  # Crop to 85% of original to make rings appear larger
        
        crop_width = int(original_width * crop_ratio)
        crop_height = int(original_height * crop_ratio)
        
        # Calculate crop box (center crop)
        left = (original_width - crop_width) // 2
        top = (original_height - crop_height) // 2
        right = left + crop_width
        bottom = top + crop_height
        
        # Crop the center
        cropped = image.crop((left, top, right, bottom))
        
        # Now resize to target dimensions
        return cropped.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
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
        
        # Use enhanced filename detection
        filename = find_filename_enhanced(event)
        if filename:
            logger.info(f"Successfully extracted filename: {filename}")
        else:
            logger.warning("Could not extract filename from input - will use default enhancement")
            # Continue processing even without filename
        
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
        
        # 1. Apply basic enhancement (matching Enhancement V111)
        enhanced_image = apply_basic_enhancement(image)
        
        # 2. Smart thumbnail creation with CENTER CROP for 90% fill
        thumbnail = create_thumbnail_smart_center_crop(enhanced_image, 1000, 1300)
        
        # 3. Apply background whitening FIRST
        thumbnail = apply_background_whitening(thumbnail)
        
        # 4. Check if unplated white based on filename
        is_unplated_white = detect_if_unplated_white(filename)
        detected_type = "무도금화이트" if is_unplated_white else "기타색상"
        logger.info(f"Final detection - Type: {detected_type}, Filename: {filename}")
        
        # 5. Apply color-specific enhancement (5% white overlay with unified brightness)
        thumbnail = apply_color_specific_enhancement(thumbnail, is_unplated_white, filename)
        
        # 6. Apply lighting effect
        thumbnail = apply_lighting_effect(thumbnail)
        
        # 7. Light sharpness for details
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
