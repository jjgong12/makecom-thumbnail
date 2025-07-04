import runpod
import os
import base64
import numpy as np
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter
import requests
import logging
import re
import replicate
import string

# Simplified imports - cv2 import moved inside function to avoid initialization issues
logging.basicConfig(level=logging.WARNING)  # Changed to WARNING to reduce logs
logger = logging.getLogger(__name__)

VERSION = "V16-EnvOptimized"

# ===== REPLICATE INITIALIZATION (환경변수 최적화) =====
REPLICATE_API_TOKEN = os.environ.get('REPLICATE_API_TOKEN')
REPLICATE_CLIENT = None
USE_REPLICATE = False

if REPLICATE_API_TOKEN:
    try:
        REPLICATE_CLIENT = replicate.Client(api_token=REPLICATE_API_TOKEN)
        USE_REPLICATE = True
        logger.info("✅ Replicate client initialized successfully for thumbnails")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Replicate client: {e}")
        USE_REPLICATE = False
else:
    logger.warning("⚠️ REPLICATE_API_TOKEN not found in environment variables")

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
    """Convert base64 to PIL Image - optimized with better error handling"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Clean the string - remove all whitespace, newlines, and special chars
        base64_string = base64_string.strip()
        base64_string = ''.join(base64_string.split())  # Remove all whitespace
        base64_string = base64_string.replace('\n', '').replace('\r', '').replace(' ', '').replace('\t', '')
        
        # Remove any non-base64 characters (keep only A-Za-z0-9+/=)
        import string
        valid_chars = string.ascii_letters + string.digits + '+/='
        base64_string = ''.join(c for c in base64_string if c in valid_chars)
        
        # Try decoding with different padding strategies
        # Strategy 1: Try as-is
        try:
            img_data = base64.b64decode(base64_string)
            return Image.open(BytesIO(img_data))
        except:
            pass
        
        # Strategy 2: Remove all padding and add correct padding
        base64_string_no_pad = base64_string.rstrip('=')
        padding = 4 - (len(base64_string_no_pad) % 4)
        if padding != 4:
            base64_string_padded = base64_string_no_pad + ('=' * padding)
        else:
            base64_string_padded = base64_string_no_pad
        
        try:
            img_data = base64.b64decode(base64_string_padded)
            return Image.open(BytesIO(img_data))
        except:
            pass
        
        # Strategy 3: Try with standard base64 (not urlsafe)
        try:
            img_data = base64.b64decode(base64_string_padded, validate=True)
            return Image.open(BytesIO(img_data))
        except:
            pass
        
        # Strategy 4: Try URL-safe base64
        try:
            # Replace standard base64 chars with URL-safe equivalents
            urlsafe_str = base64_string_no_pad.replace('+', '-').replace('/', '_')
            padding = 4 - (len(urlsafe_str) % 4)
            if padding != 4:
                urlsafe_str += '=' * padding
            img_data = base64.urlsafe_b64decode(urlsafe_str)
            return Image.open(BytesIO(img_data))
        except:
            pass
        
        # If all strategies fail, raise error
        raise ValueError("All base64 decoding strategies failed")
        
    except Exception as e:
        logger.error(f"Failed to decode base64 after all attempts: {str(e)}")
        raise ValueError(f"Failed to decode base64: {str(e)}")

def detect_pattern_type(filename: str) -> str:
    """Detect pattern type - optimized with a_ pattern"""
    if not filename:
        return "other"
    
    filename_lower = filename.lower()
    
    if 'bc_' in filename_lower:
        return "bc_only"
    elif 'b_' in filename_lower and 'bc_' not in filename_lower:
        return "b_only"
    elif 'ac_' in filename_lower:
        return "ac_only"
    elif 'a_' in filename_lower and 'ac_' not in filename_lower:
        return "a_only"
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

def apply_replicate_thumbnail_enhancement(image: Image.Image, is_wedding_ring: bool) -> Image.Image:
    """Apply Replicate enhancement for thumbnails using pre-initialized client"""
    if not USE_REPLICATE or not REPLICATE_CLIENT:
        logger.error("❌ Replicate not available for thumbnails - API token not configured")
        raise ValueError("Replicate API token not configured. Please set REPLICATE_API_TOKEN environment variable.")
    
    try:
        # Check image size and resize if necessary
        original_size = image.size
        width, height = original_size
        total_pixels = width * height
        MAX_PIXELS = 2000000  # Safe limit under 2,096,704
        
        need_resize = False
        if total_pixels > MAX_PIXELS:
            # Calculate resize factor
            resize_factor = (MAX_PIXELS / total_pixels) ** 0.5
            new_width = int(width * resize_factor)
            new_height = int(height * resize_factor)
            
            logger.info(f"⚠️ Thumbnail too large ({width}x{height}={total_pixels} pixels). Resizing to {new_width}x{new_height}")
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            need_resize = True
        
        # Convert image to base64 for Replicate
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        buffered.seek(0)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        img_data_url = f"data:image/png;base64,{img_base64}"
        
        # For thumbnails, use Real-ESRGAN for faster processing
        logger.info(f"🔷 Applying Replicate thumbnail enhancement - Wedding ring: {is_wedding_ring}")
        
        output = REPLICATE_CLIENT.run(
            "nightmareai/real-esrgan:350d32041630ffbe63c8352783a26d94126809164e54085352f8326e53999085",
            input={
                "image": img_data_url,
                "scale": 2,  # 2x upscale
                "face_enhance": False,
                "model": "RealESRGAN_x2plus"  # Faster 2x model
            }
        )
        
        if output:
            # Convert output back to PIL Image
            if isinstance(output, str):
                # If output is URL
                response = requests.get(output)
                enhanced_image = Image.open(BytesIO(response.content))
            else:
                # If output is base64 or data URL
                if hasattr(output, 'read'):
                    enhanced_image = Image.open(output)
                else:
                    enhanced_image = Image.open(BytesIO(base64.b64decode(output)))
            
            # Resize back to original size if needed
            if need_resize:
                logger.info(f"🔄 Resizing back to original size: {original_size}")
                enhanced_image = enhanced_image.resize(original_size, Image.Resampling.LANCZOS)
            
            logger.info("✅ Replicate thumbnail enhancement successful")
            return enhanced_image
        else:
            logger.error("❌ Replicate thumbnail enhancement failed - no output received")
            raise ValueError("Replicate thumbnail enhancement failed - no output received")
            
    except Exception as e:
        logger.error(f"❌ Replicate thumbnail enhancement error: {str(e)}")
        raise ValueError(f"Replicate thumbnail enhancement failed: {str(e)}")

def auto_white_balance(image: Image.Image) -> Image.Image:
    """Apply automatic white balance correction"""
    img_array = np.array(image, dtype=np.float32)
    
    # Find gray/white areas (R≈G≈B)
    gray_mask = (
        (np.abs(img_array[:,:,0] - img_array[:,:,1]) < 10) & 
        (np.abs(img_array[:,:,1] - img_array[:,:,2]) < 10) &
        (img_array[:,:,0] > 200)  # Bright areas
    )
    
    if np.sum(gray_mask) > 100:  # If enough gray pixels
        # Calculate average RGB in gray areas
        r_avg = np.mean(img_array[gray_mask, 0])
        g_avg = np.mean(img_array[gray_mask, 1])
        b_avg = np.mean(img_array[gray_mask, 2])
        
        # Calculate correction factors
        gray_avg = (r_avg + g_avg + b_avg) / 3
        r_factor = gray_avg / r_avg if r_avg > 0 else 1
        g_factor = gray_avg / g_avg if g_avg > 0 else 1
        b_factor = gray_avg / b_avg if b_avg > 0 else 1
        
        # Apply correction
        img_array[:,:,0] *= r_factor
        img_array[:,:,1] *= g_factor
        img_array[:,:,2] *= b_factor
        
        logger.info(f"White balance correction applied - R:{r_factor:.3f}, G:{g_factor:.3f}, B:{b_factor:.3f}")
    
    img_array = np.clip(img_array, 0, 255)
    return Image.fromarray(img_array.astype(np.uint8))

def correct_background_color_subtle(image: Image.Image) -> Image.Image:
    """Subtle background correction - V13 less aggressive"""
    img_array = np.array(image, dtype=np.float32)
    
    # Detect very bright background areas only
    gray = np.mean(img_array, axis=2)
    background_mask = gray > 252  # Increased threshold from 250
    
    # Make background closer to white, but not pure white
    img_array[background_mask] = np.minimum(img_array[background_mask] * 1.01, 255)  # Reduced from 1.02
    
    return Image.fromarray(img_array.astype(np.uint8))

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
    """Apply second correction for thumbnail - V13 original values with reduced cool tone"""
    if "brightness_low" in reasons:
        # Enhanced white overlay for pure white
        white_overlay_percent = 0.22  # Original value
        img_array = np.array(image, dtype=np.float32)
        img_array = img_array * (1 - white_overlay_percent) + 255 * white_overlay_percent
        img_array = np.clip(img_array, 0, 255)
        image = Image.fromarray(img_array.astype(np.uint8))
    
    if "insufficient_cool_tone" in reasons:
        img_array = np.array(image, dtype=np.float32)
        # REDUCED: Boost blue channel
        img_array[:,:,2] = np.clip(img_array[:,:,2] * 1.01, 0, 255)  # Kept reduced
        # REDUCED: Reduce red channel
        img_array[:,:,0] = np.clip(img_array[:,:,0] * 0.99, 0, 255)  # Kept reduced
        image = Image.fromarray(img_array.astype(np.uint8))
    
    if any(r in reasons for r in ["brightness_low", "saturation_high"]):
        # Apply unsharp mask
        image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=35, threshold=3))
    
    return image

def apply_center_focus_thumbnail(image: Image.Image, intensity: float = 0.025) -> Image.Image:
    """Apply subtle center focus effect for thumbnail - V13"""
    width, height = image.size
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)
    distance = np.sqrt(X**2 + Y**2)
    
    # Subtle center focus for thumbnail
    focus_mask = 1 + intensity * np.exp(-distance**2 * 2.0)
    focus_mask = np.clip(focus_mask, 1.0, 1.0 + intensity)
    
    img_array = np.array(image, dtype=np.float32)
    for i in range(3):
        img_array[:, :, i] *= focus_mask
    img_array = np.clip(img_array, 0, 255)
    
    return Image.fromarray(img_array.astype(np.uint8))

def apply_wedding_ring_focus(image: Image.Image) -> Image.Image:
    """Apply enhanced focus for wedding rings - V13 REDUCED HIGHLIGHT"""
    # 1. Highlight Enhancement - ONLY FOR NON-BACKGROUND AREAS
    img_array = np.array(image, dtype=np.float32)
    
    # Detect background (very bright and low saturation)
    gray = np.mean(img_array, axis=2)
    max_rgb = np.max(img_array, axis=2)
    min_rgb = np.min(img_array, axis=2)
    saturation = np.where(max_rgb > 0, (max_rgb - min_rgb) / max_rgb, 0)
    
    # Background mask: very bright AND low saturation
    is_background = (gray > 245) & (saturation < 0.05)
    
    # Apply bright enhancement only to non-background bright areas
    bright_mask = (img_array > 220) & (~is_background[:,:,np.newaxis])
    img_array[bright_mask] *= 1.08  # Reduced from 1.12
    img_array = np.clip(img_array, 0, 255)
    image = Image.fromarray(img_array.astype(np.uint8))
    
    # 2. Center focus - 4.5% (RESTORED)
    width, height = image.size
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)
    distance = np.sqrt(X**2 + Y**2)
    
    # 4.5% center focus (RESTORED)
    focus_mask = 1 + 0.045 * np.exp(-distance**2 * 1.5)
    focus_mask = np.clip(focus_mask, 1.0, 1.045)
    
    img_array = np.array(image, dtype=np.float32)
    for i in range(3):
        img_array[:, :, i] *= focus_mask
    img_array = np.clip(img_array, 0, 255)
    image = Image.fromarray(img_array.astype(np.uint8))
    
    # 3. Enhanced sharpness (KEPT SAME)
    sharpness = ImageEnhance.Sharpness(image)
    image = sharpness.enhance(1.35)
    
    # 4. Enhanced Contrast (KEPT SAME)
    contrast = ImageEnhance.Contrast(image)
    image = contrast.enhance(1.04)
    
    # 5. Brightness enhancement (REDUCED)
    brightness = ImageEnhance.Brightness(image)
    image = brightness.enhance(1.02)  # Reduced from 1.03
    
    # 6. Structure Enhancement (KEPT SAME)
    image = image.filter(ImageFilter.UnsharpMask(radius=1.0, percent=60, threshold=2))
    
    # 7. Micro Contrast (KEPT SAME)
    gray = image.convert('L')
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edges_array = np.array(edges, dtype=np.float32) * 0.08  # 8% micro contrast
    
    img_array = np.array(image, dtype=np.float32)
    for i in range(3):
        img_array[:, :, i] += edges_array
    img_array = np.clip(img_array, 0, 255)
    image = Image.fromarray(img_array.astype(np.uint8))
    
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
    
    # Apply white balance correction FIRST
    image = auto_white_balance(image)
    
    # Enhanced brightness - RESTORED to 1.025
    brightness = ImageEnhance.Brightness(image)
    image = brightness.enhance(1.025)
    
    contrast = ImageEnhance.Contrast(image)
    image = contrast.enhance(1.03)
    
    color = ImageEnhance.Color(image)
    image = color.enhance(1.02)
    
    return image

def apply_pattern_enhancement(image, pattern_type, is_wedding_ring):
    """Apply enhancement based on pattern type - V16 with 5% overlay for a/b patterns"""
    
    if pattern_type == "bc_only":
        # bc_ pattern (unplated white) - V13 original values
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.04)  # RESTORED
        
        color = ImageEnhance.Color(image)
        image = color.enhance(0.95)  # More desaturated for pure white
        
        # V13: Original white overlay for bc_ - 0.14
        white_overlay = 0.14  # RESTORED
        img_array = np.array(image, dtype=np.float32)
        img_array = img_array * (1 - white_overlay) + 255 * white_overlay
        img_array = np.clip(img_array, 0, 255)
        image = Image.fromarray(img_array.astype(np.uint8))
        
        # V13: 7% center focus (RESTORED)
        width, height = image.size
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        X, Y = np.meshgrid(x, y)
        distance = np.sqrt(X**2 + Y**2)
        
        focus_mask = 1 + 0.07 * np.exp(-distance**2 * 1.5)
        focus_mask = np.clip(focus_mask, 1.0, 1.07)
        
        img_array = np.array(image, dtype=np.float32)
        for i in range(3):
            img_array[:, :, i] *= focus_mask
        img_array = np.clip(img_array, 0, 255)
        image = Image.fromarray(img_array.astype(np.uint8))
        
        # V13: Add subtle center focus
        image = apply_center_focus_thumbnail(image, 0.025)
        
        if is_wedding_ring:
            image = apply_wedding_ring_focus(image)
            
    elif pattern_type == "ac_only":
        # ac_ pattern (unplated white) - same as bc_
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.04)
        
        color = ImageEnhance.Color(image)
        image = color.enhance(0.95)
        
        # ac_ also uses 0.14 white overlay
        white_overlay = 0.14
        img_array = np.array(image, dtype=np.float32)
        img_array = img_array * (1 - white_overlay) + 255 * white_overlay
        img_array = np.clip(img_array, 0, 255)
        image = Image.fromarray(img_array.astype(np.uint8))
        
        # 7% center focus
        width, height = image.size
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        X, Y = np.meshgrid(x, y)
        distance = np.sqrt(X**2 + Y**2)
        
        focus_mask = 1 + 0.07 * np.exp(-distance**2 * 1.5)
        focus_mask = np.clip(focus_mask, 1.0, 1.07)
        
        img_array = np.array(image, dtype=np.float32)
        for i in range(3):
            img_array[:, :, i] *= focus_mask
        img_array = np.clip(img_array, 0, 255)
        image = Image.fromarray(img_array.astype(np.uint8))
        
        image = apply_center_focus_thumbnail(image, 0.025)
        
        if is_wedding_ring:
            image = apply_wedding_ring_focus(image)
        
    elif pattern_type == "b_only":
        # b_ pattern - V16 UPDATED to 5%
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.04)  # RESTORED
        
        color = ImageEnhance.Color(image)
        image = color.enhance(0.95)  # Same desaturation as bc_
        
        # V16: UPDATED white overlay for b_ - 0.05 (5%)
        white_overlay = 0.05  # CHANGED from 0.06 to 0.05
        img_array = np.array(image, dtype=np.float32)
        img_array = img_array * (1 - white_overlay) + 255 * white_overlay
        img_array = np.clip(img_array, 0, 255)
        image = Image.fromarray(img_array.astype(np.uint8))
        
        # Enhanced sharpness for b_ pattern
        sharpness = ImageEnhance.Sharpness(image)
        image = sharpness.enhance(1.15)
        
        # V13: 7% center focus (RESTORED)
        width, height = image.size
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        X, Y = np.meshgrid(x, y)
        distance = np.sqrt(X**2 + Y**2)
        
        focus_mask = 1 + 0.07 * np.exp(-distance**2 * 1.0)
        focus_mask = np.clip(focus_mask, 1.0, 1.07)
        
        img_array = np.array(image, dtype=np.float32)
        for i in range(3):
            img_array[:, :, i] *= focus_mask
        img_array = np.clip(img_array, 0, 255)
        image = Image.fromarray(img_array.astype(np.uint8))
        
        # V13: Add subtle center focus
        image = apply_center_focus_thumbnail(image, 0.025)
        
        if is_wedding_ring:
            image = apply_wedding_ring_focus(image)
            
    elif pattern_type == "a_only":
        # a_ pattern - V16 NEW with 5%
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.04)
        
        color = ImageEnhance.Color(image)
        image = color.enhance(0.95)
        
        # V16: NEW white overlay for a_ - 0.05 (5%)
        white_overlay = 0.05
        img_array = np.array(image, dtype=np.float32)
        img_array = img_array * (1 - white_overlay) + 255 * white_overlay
        img_array = np.clip(img_array, 0, 255)
        image = Image.fromarray(img_array.astype(np.uint8))
        
        # Enhanced sharpness for a_ pattern
        sharpness = ImageEnhance.Sharpness(image)
        image = sharpness.enhance(1.15)
        
        # 7% center focus
        width, height = image.size
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        X, Y = np.meshgrid(x, y)
        distance = np.sqrt(X**2 + Y**2)
        
        focus_mask = 1 + 0.07 * np.exp(-distance**2 * 1.0)
        focus_mask = np.clip(focus_mask, 1.0, 1.07)
        
        img_array = np.array(image, dtype=np.float32)
        for i in range(3):
            img_array[:, :, i] *= focus_mask
        img_array = np.clip(img_array, 0, 255)
        image = Image.fromarray(img_array.astype(np.uint8))
        
        image = apply_center_focus_thumbnail(image, 0.025)
        
        if is_wedding_ring:
            image = apply_wedding_ring_focus(image)
        
    else:
        # Standard enhancement
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.025)
        
        color = ImageEnhance.Color(image)
        image = color.enhance(0.98)
        
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.02)
        
        # V13: Add subtle center focus to other patterns too
        image = apply_center_focus_thumbnail(image, 0.02)
        
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

def needs_thumbnail_upscaling(image: Image.Image) -> bool:
    """Check if thumbnail needs upscaling"""
    width, height = image.size
    # Thumbnail needs upscaling if source is smaller than 2000x2600
    return width < 2000 or height < 2600

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
    """Thumbnail handler function - V16 with environment variable optimization"""
    try:
        logger.info(f"=== Thumbnail {VERSION} Started ===")
        logger.info(f"Replicate available: {USE_REPLICATE}")
        
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
        
        # Apply basic enhancement (includes white balance)
        enhanced_image = apply_basic_enhancement(image)
        
        # Detect wedding ring
        is_wedding_ring = detect_wedding_ring_fast(enhanced_image)
        logger.info(f"Wedding ring detected: {is_wedding_ring}")
        
        # Check if upscaling is needed
        needs_upscale = needs_thumbnail_upscaling(enhanced_image)
        replicate_applied = False
        replicate_resized = False
        
        # Apply Replicate enhancement if available (웨딩링이므로 항상 적용)
        if USE_REPLICATE:
            logger.info(f"Applying Replicate thumbnail enhancement - Wedding ring: {is_wedding_ring}, Needs upscale: {needs_upscale}")
            try:
                original_replicate_size = enhanced_image.size
                enhanced_image = apply_replicate_thumbnail_enhancement(enhanced_image, is_wedding_ring)
                replicate_applied = True
                # Check if image was resized for Replicate
                total_pixels = original_replicate_size[0] * original_replicate_size[1]
                replicate_resized = total_pixels > 2000000
            except Exception as e:
                logger.error(f"Replicate thumbnail enhancement failed: {str(e)}")
                return {
                    "output": {
                        "error": f"Replicate thumbnail enhancement failed: {str(e)}",
                        "status": "failed",
                        "version": VERSION
                    }
                }
        
        # Create thumbnail with upscaling
        thumbnail = create_thumbnail_smart_center_crop_with_upscale(enhanced_image, 1000, 1300)
        
        # Detect pattern type
        pattern_type = detect_pattern_type(filename)
        
        if pattern_type == "bc_only":
            detected_type = "무도금화이트(0.14)"
        elif pattern_type == "ac_only":
            detected_type = "무도금화이트(0.14)"
        elif pattern_type == "b_only":
            detected_type = "b_패턴(0.05)"
        elif pattern_type == "a_only":
            detected_type = "a_패턴(0.05)"
        else:
            detected_type = "기타색상"
        
        # Apply pattern-specific enhancement
        thumbnail = apply_pattern_enhancement(thumbnail, pattern_type, is_wedding_ring)
        
        # Quality check for bc_only pattern ONLY - V13 standards
        second_correction_applied = False
        if pattern_type == "bc_only":  # Only bc_only pattern
            metrics = calculate_quality_metrics_simple(thumbnail)
            
            reasons = []
            if metrics["brightness"] < 241:
                reasons.append("brightness_low")
            if metrics["cool_tone_diff"] < 3:
                reasons.append("insufficient_cool_tone")
            if metrics["rgb_deviation"] > 5:
                reasons.append("rgb_deviation_high")
            if metrics["saturation"] > 2:
                reasons.append("saturation_high")
            
            if reasons:
                thumbnail = apply_second_correction_thumbnail(thumbnail, reasons)
                second_correction_applied = True
        
        # Apply spotlight (only if not wedding ring)
        if not is_wedding_ring:
            thumbnail = apply_spotlight_effect(thumbnail)
        
        # V13: Enhanced sharpness - differentiated by wedding ring
        sharpness = ImageEnhance.Sharpness(thumbnail)
        if is_wedding_ring:
            thumbnail = sharpness.enhance(1.25)
        else:
            thumbnail = sharpness.enhance(1.4)
        
        # Final brightness touch - REDUCED for background safety
        brightness = ImageEnhance.Brightness(thumbnail)
        thumbnail = brightness.enhance(1.02)  # Reduced from 1.04
        
        # Apply SUBTLE background correction for final touch
        thumbnail = correct_background_color_subtle(thumbnail)
        
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
                "replicate_enhancement": {
                    "applied": replicate_applied,
                    "upscaling_needed": needs_upscale,
                    "available": USE_REPLICATE,
                    "input_resized_for_gpu": replicate_resized if replicate_applied else None,
                    "model_used": "real-esrgan-x2plus"
                },
                "white_overlay_info": {
                    "bc_only": "0.14",
                    "ac_only": "0.14",
                    "b_only": "0.05",
                    "a_only": "0.05",
                    "other": "none"
                },
                "has_center_focus": True,
                "center_focus_intensity": "7%",
                "white_balance_applied": True,
                "cool_tone_reduced": True,
                "background_correction": "subtle",
                "wedding_ring_enhancements": {
                    "highlight_enhancement": "8%",
                    "highlight_threshold": "220",
                    "micro_contrast": "8%",
                    "structure_enhancement": "enabled",
                    "enhanced_sharpness": "1.35",
                    "enhanced_contrast": "1.04",
                    "enhanced_brightness": "1.02",
                    "enhanced_center_focus": "4.5%",
                    "final_sharpness": "1.25"
                } if is_wedding_ring else None
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
