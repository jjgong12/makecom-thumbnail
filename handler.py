import runpod
import base64
import numpy as np
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter
import requests
import logging
import re

# Simplified imports - cv2 import moved inside function to avoid initialization issues
logging.basicConfig(level=logging.WARNING)  # Changed to WARNING to reduce logs
logger = logging.getLogger(__name__)

VERSION = "V9-WhiteOverlay-Extended"

def find_input_data(data):
    """Find input data recursively - optimized"""
    if isinstance(data, dict):
        # Check direct image keys
        for key in ['image', 'image_base64', 'imageBase64', 'url', 'image_url']:
            if key in data:
                return data
        
        # Check nested structures
        for key in ['input', 'job', 'payload', 'data']:
            if key in data:
                result = find_input_data(data[key])
                if result:
                    return result
    return data

def find_filename_optimized(data, depth=0):
    """Optimized filename extraction"""
    if depth > 5:
        return None
    
    if isinstance(data, dict):
        # Direct filename keys
        for key in ['filename', 'file_name', 'fileName', 'name', 'originalName']:
            if key in data and isinstance(data[key], str):
                value = data[key]
                if any(p in value.lower() for p in ['b_', 'bc_', 'a_', 'ac_']):
                    return value
                elif '.' in value and len(value) < 100:
                    return value
        
        # Recursive search
        for value in data.values():
            if isinstance(value, dict):
                result = find_filename_optimized(value, depth + 1)
                if result:
                    return result
    
    return None

def generate_thumbnail_filename(original_filename, image_index):
    """Generate thumbnail filename with 007, 008, 009 pattern"""
    if not original_filename:
        return f"thumbnail_{image_index:03d}.jpg"
    
    # Map image index to 007, 008, 009
    thumbnail_numbers = {
        1: "007",
        3: "008",
        5: "009"
    }
    
    new_filename = original_filename
    pattern = r'(_\d{3})'
    if re.search(pattern, new_filename):
        new_filename = re.sub(pattern, f'_{thumbnail_numbers.get(image_index, "007")}', new_filename)
    else:
        name_parts = new_filename.split('.')
        name_parts[0] += f'_{thumbnail_numbers.get(image_index, "007")}'
        new_filename = '.'.join(name_parts)
    
    return new_filename

def download_image_from_url(url):
    """Download image from URL - with error handling"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        logger.error(f"Failed to download image: {str(e)}")
        raise

def base64_to_image(base64_string):
    """Convert base64 to PIL Image - optimized"""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Clean and add padding
        base64_string = base64_string.strip()
        padding = 4 - len(base64_string) % 4
        if padding != 4:
            base64_string += '=' * padding
        
        img_data = base64.b64decode(base64_string)
        return Image.open(BytesIO(img_data))
    except Exception as e:
        logger.error(f"Failed to decode base64: {str(e)}")
        raise

def detect_pattern_type(filename: str) -> str:
    """Detect pattern type - optimized"""
    if not filename:
        return "other"
    
    filename_lower = filename.lower()
    
    if 'bc_' in filename_lower:
        return "bc_only"
    elif 'b_' in filename_lower and 'bc_' not in filename_lower:
        return "b_only"
    else:
        return "other"

def detect_wedding_ring_fast(image: Image.Image) -> bool:
    """Fast wedding ring detection - simplified without cv2"""
    try:
        # Check center region for bright metallic areas
        width, height = image.size
        center_crop = image.crop((width//3, height//3, 2*width//3, 2*height//3))
        gray = center_crop.convert('L')
        gray_array = np.array(gray)
        
        # Check bright pixels ratio
        bright_pixels = np.sum(gray_array > 200)
        total_pixels = gray_array.size
        bright_ratio = bright_pixels / total_pixels
        
        return bool(bright_ratio > 0.15)
    except:
        return False

def calculate_quality_metrics_simple(image: Image.Image) -> dict:
    """Calculate quality metrics without cv2"""
    img_array = np.array(image)
    
    # Calculate RGB averages
    r_avg = float(np.mean(img_array[:,:,0]))
    g_avg = float(np.mean(img_array[:,:,1]))
    b_avg = float(np.mean(img_array[:,:,2]))
    
    # Calculate brightness
    brightness = (r_avg + g_avg + b_avg) / 3
    
    # Calculate RGB deviation
    rgb_values = [r_avg, g_avg, b_avg]
    rgb_deviation = max(rgb_values) - min(rgb_values)
    
    # Simple saturation calculation without cv2
    # Using max-min method for each pixel
    max_rgb = np.max(img_array, axis=2)
    min_rgb = np.min(img_array, axis=2)
    diff = max_rgb - min_rgb
    # Avoid division by zero
    saturation = float(np.mean(np.where(max_rgb > 0, diff / max_rgb, 0)) * 100)
    
    # Cool tone check
    cool_tone_diff = b_avg - r_avg
    
    return {
        "brightness": brightness,
        "rgb_deviation": rgb_deviation,
        "saturation": saturation,
        "cool_tone_diff": cool_tone_diff
    }

def apply_second_correction_thumbnail(image: Image.Image, reasons: list) -> Image.Image:
    """Apply second correction for thumbnail - V9 pure white"""
    if "brightness_low" in reasons:
        # Enhanced white overlay for pure white
        white_overlay_percent = 0.22  # Increased for thumbnail
        img_array = np.array(image, dtype=np.float32)
        img_array = img_array * (1 - white_overlay_percent) + 255 * white_overlay_percent
        img_array = np.clip(img_array, 0, 255)
        image = Image.fromarray(img_array.astype(np.uint8))
    
    if "insufficient_cool_tone" in reasons:
        img_array = np.array(image, dtype=np.float32)
        # Boost blue channel
        img_array[:,:,2] = np.clip(img_array[:,:,2] * 1.03, 0, 255)
        # Reduce red channel
        img_array[:,:,0] = np.clip(img_array[:,:,0] * 0.97, 0, 255)
        image = Image.fromarray(img_array.astype(np.uint8))
    
    if any(r in reasons for r in ["brightness_low", "saturation_high"]):
        # Apply unsharp mask
        image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=35, threshold=3))
    
    return image

def apply_wedding_ring_focus(image: Image.Image) -> Image.Image:
    """Apply enhanced focus for wedding rings"""
    # Center focus
    width, height = image.size
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)
    distance = np.sqrt(X**2 + Y**2)
    
    # 3.5% center focus
    focus_mask = 1 + 0.035 * np.exp(-distance**2 * 1.5)
    focus_mask = np.clip(focus_mask, 1.0, 1.035)
    
    img_array = np.array(image, dtype=np.float32)
    for i in range(3):
        img_array[:, :, i] *= focus_mask
    img_array = np.clip(img_array, 0, 255)
    image = Image.fromarray(img_array.astype(np.uint8))
    
    # Enhanced sharpness
    sharpness = ImageEnhance.Sharpness(image)
    image = sharpness.enhance(1.3)
    
    # Contrast
    contrast = ImageEnhance.Contrast(image)
    image = contrast.enhance(1.03)
    
    return image

def apply_basic_enhancement(image):
    """Apply basic enhancement"""
    if image.mode != 'RGB':
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        else:
            image = image.convert('RGB')
    
    # Enhanced brightness
    brightness = ImageEnhance.Brightness(image)
    image = brightness.enhance(1.05)
    
    contrast = ImageEnhance.Contrast(image)
    image = contrast.enhance(1.03)
    
    color = ImageEnhance.Color(image)
    image = color.enhance(1.02)
    
    return image

def apply_pattern_enhancement(image, pattern_type, is_wedding_ring):
    """Apply enhancement based on pattern type - V9 with b_ white overlay"""
    
    if pattern_type == "bc_only":
        # bc_ pattern (unplated white) - V8 adjustments
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.01)  # Lowered from 1.03
        
        color = ImageEnhance.Color(image)
        image = color.enhance(0.95)  # More desaturated for pure white
        
        # Increased white overlay for pure white effect
        white_overlay = 0.20 if is_wedding_ring else 0.17  # Increased by 2%
        img_array = np.array(image, dtype=np.float32)
        img_array = img_array * (1 - white_overlay) + 255 * white_overlay
        img_array = np.clip(img_array, 0, 255)
        image = Image.fromarray(img_array.astype(np.uint8))
        
        if is_wedding_ring:
            image = apply_wedding_ring_focus(image)
        
    elif pattern_type == "b_only":
        # b_ pattern - V9 WITH WHITE OVERLAY
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.01)  # Same as bc_
        
        color = ImageEnhance.Color(image)
        image = color.enhance(0.95)  # Same desaturation as bc_
        
        # V9: ADD WHITE OVERLAY TO b_ PATTERN (same as bc_)
        white_overlay = 0.20 if is_wedding_ring else 0.17
        img_array = np.array(image, dtype=np.float32)
        img_array = img_array * (1 - white_overlay) + 255 * white_overlay
        img_array = np.clip(img_array, 0, 255)
        image = Image.fromarray(img_array.astype(np.uint8))
        
        # Center focus
        width, height = image.size
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        X, Y = np.meshgrid(x, y)
        distance = np.sqrt(X**2 + Y**2)
        
        focus_mask = 1 + 0.03 * np.exp(-distance**2 * 1.0)
        focus_mask = np.clip(focus_mask, 1.0, 1.03)
        
        img_array = np.array(image, dtype=np.float32)
        for i in range(3):
            img_array[:, :, i] *= focus_mask
        img_array = np.clip(img_array, 0, 255)
        image = Image.fromarray(img_array.astype(np.uint8))
        
        if is_wedding_ring:
            image = apply_wedding_ring_focus(image)
        
    else:
        # Standard enhancement
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.03)
        
        color = ImageEnhance.Color(image)
        image = color.enhance(0.98)
        
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.02)
        
        if is_wedding_ring:
            image = apply_wedding_ring_focus(image)
    
    return image

def apply_spotlight_effect(image):
    """Apply subtle spotlight effect"""
    width, height = image.size
    
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)
    distance = np.sqrt(X**2 + Y**2)
    
    # Spotlight mask
    spotlight_mask = 1 + 0.03 * np.exp(-distance**2 * 1.2)
    spotlight_mask = np.clip(spotlight_mask, 1.0, 1.03)
    
    img_array = np.array(image, dtype=np.float32)
    for i in range(3):
        img_array[:, :, i] *= spotlight_mask
    img_array = np.clip(img_array, 0, 255)
    
    return Image.fromarray(img_array.astype(np.uint8))

def create_thumbnail_smart_center_crop_with_upscale(image, target_width=1000, target_height=1300):
    """Create thumbnail with fixed center crop and upscaling"""
    original_width, original_height = image.size
    
    # Fixed image center
    image_center = (original_width // 2, original_height // 2)
    
    # Upscale if needed
    if original_width < target_width or original_height < target_height:
        scale_factor = max(target_width / original_width, target_height / original_height) * 1.1
        
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        image_center = (int(image_center[0] * scale_factor), int(image_center[1] * scale_factor))
        original_width, original_height = new_width, new_height
    
    # Check for specific size ranges
    if ((1800 <= original_width <= 2200 and 2400 <= original_height <= 2800) or
        (2800 <= original_width <= 3200 and 3700 <= original_height <= 4100)):
        
        crop_ratio = 0.75 if original_width >= 2800 else 0.85
        
        crop_width = int(original_width * crop_ratio)
        crop_height = int(original_height * crop_ratio)
        
        # Fixed center crop
        left = max(0, image_center[0] - crop_width // 2)
        top = max(0, image_center[1] - crop_height // 2)
        
        if left + crop_width > original_width:
            left = original_width - crop_width
        if top + crop_height > original_height:
            top = original_height - crop_height
        
        right = left + crop_width
        bottom = top + crop_height
        
        cropped = image.crop((left, top, right, bottom))
        
        return cropped.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
    else:
        # Aspect ratio preservation
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
    """Convert image to base64 without padding"""
    buffered = BytesIO()
    
    if image.mode == 'RGBA':
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
    
    image.save(buffered, format='PNG', quality=95)
    buffered.seek(0)  # Important!
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    return img_base64.rstrip('=')

def handler(event):
    """Thumbnail handler function - V9 with b_ white overlay"""
    try:
        # Get image index
        image_index = event.get('image_index', 1)
        if isinstance(event.get('input'), dict):
            image_index = event.get('input', {}).get('image_index', image_index)
        
        # Extract filename
        filename = find_filename_optimized(event)
        logger.info(f"Processing thumbnail for: {filename}")
        
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
            elif 'url' in input_data or 'image_url' in input_data:
                image_url = input_data.get('url') or input_data.get('image_url')
                image = download_image_from_url(image_url)
        elif isinstance(input_data, str):
            if input_data.startswith('http'):
                image = download_image_from_url(input_data)
            else:
                image = base64_to_image(input_data)
        
        if not image:
            raise ValueError("Failed to load image")
        
        # Apply basic enhancement
        enhanced_image = apply_basic_enhancement(image)
        
        # Detect wedding ring
        is_wedding_ring = detect_wedding_ring_fast(enhanced_image)
        
        # Create thumbnail with upscaling
        thumbnail = create_thumbnail_smart_center_crop_with_upscale(enhanced_image, 1000, 1300)
        
        # Detect pattern type
        pattern_type = detect_pattern_type(filename)
        
        if pattern_type == "bc_only":
            detected_type = "무도금화이트"
        elif pattern_type == "b_only":
            detected_type = "b_패턴(화이트오버레이)"  # V9: Now with white overlay
        else:
            detected_type = "기타색상"
        
        # Apply pattern-specific enhancement
        thumbnail = apply_pattern_enhancement(thumbnail, pattern_type, is_wedding_ring)
        
        # Quality check for bc_ and b_ patterns - V9 stricter standards
        second_correction_applied = False
        if pattern_type in ["bc_only", "b_only"]:  # V9: Extended to b_only
            metrics = calculate_quality_metrics_simple(thumbnail)
            
            reasons = []
            if metrics["brightness"] < 241:  # Increased from 235
                reasons.append("brightness_low")
            if metrics["cool_tone_diff"] < 3:
                reasons.append("insufficient_cool_tone")
            if metrics["rgb_deviation"] > 5:
                reasons.append("rgb_deviation_high")
            if metrics["saturation"] > 2:  # Reduced from 3
                reasons.append("saturation_high")
            
            if reasons:
                thumbnail = apply_second_correction_thumbnail(thumbnail, reasons)
                second_correction_applied = True
        
        # Apply spotlight (only if not wedding ring)
        if not is_wedding_ring:
            thumbnail = apply_spotlight_effect(thumbnail)
        
        # Enhanced sharpness
        sharpness = ImageEnhance.Sharpness(thumbnail)
        if is_wedding_ring:
            thumbnail = sharpness.enhance(1.2)
        else:
            thumbnail = sharpness.enhance(1.35)
        
        # Final brightness touch
        brightness = ImageEnhance.Brightness(thumbnail)
        thumbnail = brightness.enhance(1.05)
        
        # Convert to base64
        thumbnail_base64 = image_to_base64(thumbnail)
        
        # Generate output filename
        output_filename = generate_thumbnail_filename(filename, image_index)
        
        # Return result
        return {
            "output": {
                "thumbnail": thumbnail_base64,
                "size": list(thumbnail.size),
                "detected_type": detected_type,
                "pattern_type": pattern_type,
                "is_wedding_ring": bool(is_wedding_ring),
                "filename": output_filename,
                "original_filename": filename,
                "image_index": image_index,
                "format": "base64_no_padding",
                "has_spotlight": not is_wedding_ring,
                "has_upscaling": True,
                "second_correction_applied": bool(second_correction_applied),
                "version": VERSION,
                "status": "success",
                "white_overlay_info": {
                    "bc_only": "0.17-0.20",
                    "b_only": "0.17-0.20",  # V9: Now same as bc_only
                    "other": "none"
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error in handler: {str(e)}")
        import traceback
        
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
