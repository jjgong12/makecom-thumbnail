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

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

VERSION = "V19-Bright-Enhanced-Failsafe"

# ===== REPLICATE INITIALIZATION =====
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

def find_input_data(data):
    """Find input data recursively - ULTRA ENHANCED for Make.com"""
    if isinstance(data, dict):
        # Extended image keys
        image_keys = [
            'image', 'image_base64', 'imageBase64', 'url', 'image_url', 
            'enhanced_image', 'thumbnail', 'base64', 'img', 'photo',
            'picture', 'file', 'content', 'b64', 'base64_data',
            'data', 'raw_image', 'image_content'
        ]
        
        # Check direct keys first
        for key in image_keys:
            if key in data:
                value = data[key]
                if isinstance(value, str):
                    if value.startswith('http') and key in ['url', 'image_url']:
                        return data
                    elif len(value) > 50:
                        return data
        
        # Check nested structures
        nested_keys = ['input', 'job', 'payload', 'data', 'body', 'request', 
                      'inputs', 'params', 'arguments', 'output']
        for key in nested_keys:
            if key in data:
                result = find_input_data(data[key])
                if result:
                    return result
        
        # Check numeric keys (Make.com)
        for i in range(20):
            str_key = str(i)
            if str_key in data:
                if isinstance(data[str_key], dict):
                    result = find_input_data(data[str_key])
                    if result:
                        return result
                elif isinstance(data[str_key], str) and len(data[str_key]) > 50:
                    return {image_keys[0]: data[str_key]}
                
    return data

def find_filename_optimized(data, depth=0):
    """Optimized filename extraction"""
    if depth > 5:
        return None
    
    if isinstance(data, dict):
        for key in ['filename', 'file_name', 'fileName', 'name', 'originalName']:
            if key in data and isinstance(data[key], str):
                value = data[key]
                if any(p in value.lower() for p in ['b_', 'bc_', 'a_', 'ac_']):
                    return value
                elif '.' in value and len(value) < 100:
                    return value
        
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
    """Download image from URL"""
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

def base64_to_image_ultra_safe(base64_string):
    """Convert base64 to PIL Image - V19 ULTRA SAFE"""
    try:
        if not base64_string:
            raise ValueError("Empty base64 string")
            
        if isinstance(base64_string, bytes):
            return Image.open(BytesIO(base64_string))
            
        base64_string = str(base64_string)
        
        # AGGRESSIVE cleaning
        # 1. Remove ALL possible prefixes
        if 'base64,' in base64_string:
            base64_string = base64_string.split('base64,')[-1]
        elif 'data:' in base64_string:
            parts = base64_string.split(',')
            if len(parts) > 1:
                base64_string = parts[-1]
            else:
                base64_string = base64_string.split('base64')[-1].lstrip(',')
        
        # 2. ULTRA clean
        base64_string = base64_string.strip()
        base64_string = ''.join(base64_string.split())
        
        # 3. Handle encoding issues
        replacements = [
            ('%2B', '+'), ('%2F', '/'), ('%3D', '='),
            ('%2b', '+'), ('%2f', '/'), ('%3d', '='),
            (' ', ''), ('\n', ''), ('\r', ''), ('\t', ''),
            ('\\n', ''), ('\\r', ''), ('\\t', ''),
            ('"', ''), ("'", '')
        ]
        
        for old, new in replacements:
            base64_string = base64_string.replace(old, new)
        
        # 4. Keep ONLY valid base64 characters
        valid_chars = set(string.ascii_letters + string.digits + '+/=')
        base64_string = ''.join(c for c in base64_string if c in valid_chars)
        
        # 5. Check minimum length
        if len(base64_string) < 50:
            raise ValueError(f"Base64 too short: {len(base64_string)} chars")
        
        # 6. Try multiple padding strategies
        strategies = []
        no_pad = base64_string.rstrip('=')
        
        # Strategy order (Make.com compatible):
        strategies.append(no_pad)  # No padding (Make.com style)
        padding_needed = (4 - len(no_pad) % 4) % 4
        strategies.append(no_pad + ('=' * padding_needed))  # Correct padding
        if base64_string != no_pad:
            strategies.append(base64_string)  # Original
        for i in range(4):
            padded = no_pad + ('=' * i)
            if padded not in strategies:
                strategies.append(padded)
        
        # 7. Try each strategy
        last_error = None
        for i, test_string in enumerate(strategies):
            if not test_string:
                continue
            
            try:
                img_data = base64.b64decode(test_string, validate=True)
                img = Image.open(BytesIO(img_data))
                logger.debug(f"✅ Decoded with strategy {i} (standard)")
                return img
            except Exception as e:
                last_error = e
            
            try:
                urlsafe = test_string.replace('+', '-').replace('/', '_')
                img_data = base64.urlsafe_b64decode(urlsafe)
                img = Image.open(BytesIO(img_data))
                logger.debug(f"✅ Decoded with strategy {i} (urlsafe)")
                return img
            except:
                pass
            
            try:
                img_data = base64.b64decode(test_string, validate=False)
                img = Image.open(BytesIO(img_data))
                logger.debug(f"✅ Decoded with strategy {i} (no validation)")
                return img
            except:
                pass
        
        logger.error(f"❌ All decode attempts failed. Length: {len(base64_string)}")
        raise ValueError(f"Base64 decode failed: {last_error}")
        
    except Exception as e:
        logger.error(f"Base64 decode error: {str(e)}")
        raise ValueError(f"Invalid base64 data: {str(e)}")

def detect_pattern_type(filename: str) -> str:
    """Detect pattern type"""
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
    """Always return True since all images are wedding rings"""
    return True

def apply_replicate_thumbnail_enhancement(image: Image.Image, is_wedding_ring: bool) -> Image.Image:
    """Apply Replicate enhancement for thumbnails"""
    if not USE_REPLICATE or not REPLICATE_CLIENT:
        logger.error("❌ Replicate not available for thumbnails")
        raise ValueError("Replicate API token not configured")
    
    try:
        # Check image size
        original_size = image.size
        width, height = original_size
        total_pixels = width * height
        MAX_PIXELS = 2000000
        
        need_resize = False
        if total_pixels > MAX_PIXELS:
            resize_factor = (MAX_PIXELS / total_pixels) ** 0.5
            new_width = int(width * resize_factor)
            new_height = int(height * resize_factor)
            
            logger.info(f"Resizing for Replicate: {width}x{height} -> {new_width}x{new_height}")
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            need_resize = True
        
        # Convert to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        buffered.seek(0)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        img_data_url = f"data:image/png;base64,{img_base64}"
        
        logger.info(f"🔷 Applying Replicate thumbnail enhancement with 2x upscaling")
        
        # Use Real-ESRGAN with 2x upscaling (reduced from 4x)
        output = REPLICATE_CLIENT.run(
            "nightmareai/real-esrgan:350d32041630ffbe63c8352783a26d94126809164e54085352f8326e53999085",
            input={
                "image": img_data_url,
                "scale": 2,  # Reduced from 4 to 2
                "face_enhance": False,
                "model": "RealESRGAN_x4plus"
            }
        )
        
        if output:
            # Convert output back
            if isinstance(output, str):
                response = requests.get(output)
                enhanced_image = Image.open(BytesIO(response.content))
            else:
                if hasattr(output, 'read'):
                    enhanced_image = Image.open(output)
                else:
                    enhanced_image = Image.open(BytesIO(base64.b64decode(output)))
            
            if need_resize:
                logger.info(f"Resizing back to original: {original_size}")
                enhanced_image = enhanced_image.resize(original_size, Image.Resampling.LANCZOS)
            
            logger.info("✅ Replicate thumbnail enhancement successful")
            return enhanced_image
        else:
            raise ValueError("Replicate thumbnail enhancement failed")
            
    except Exception as e:
        logger.error(f"❌ Replicate thumbnail enhancement error: {str(e)}")
        raise

def auto_white_balance(image: Image.Image) -> Image.Image:
    """Apply automatic white balance correction"""
    img_array = np.array(image, dtype=np.float32)
    
    gray_mask = (
        (np.abs(img_array[:,:,0] - img_array[:,:,1]) < 10) & 
        (np.abs(img_array[:,:,1] - img_array[:,:,2]) < 10) &
        (img_array[:,:,0] > 200)
    )
    
    if np.sum(gray_mask) > 100:
        r_avg = np.mean(img_array[gray_mask, 0])
        g_avg = np.mean(img_array[gray_mask, 1])
        b_avg = np.mean(img_array[gray_mask, 2])
        
        gray_avg = (r_avg + g_avg + b_avg) / 3
        r_factor = gray_avg / r_avg if r_avg > 0 else 1
        g_factor = gray_avg / g_avg if g_avg > 0 else 1
        b_factor = gray_avg / b_avg if b_avg > 0 else 1
        
        img_array[:,:,0] *= r_factor
        img_array[:,:,1] *= g_factor
        img_array[:,:,2] *= b_factor
    
    img_array = np.clip(img_array, 0, 255)
    return Image.fromarray(img_array.astype(np.uint8))

def apply_center_spotlight_thumbnail(image: Image.Image, intensity: float = 0.10) -> Image.Image:
    """Apply center spotlight for thumbnail - Reduced for wedding rings"""
    width, height = image.size
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)
    distance = np.sqrt(X**2 + Y**2)
    
    # Reduced spotlight
    spotlight_mask = 1 + intensity * np.exp(-distance**2 * 0.8)
    spotlight_mask = np.clip(spotlight_mask, 1.0, 1.0 + intensity)
    
    img_array = np.array(image, dtype=np.float32)
    for i in range(3):
        img_array[:, :, i] *= spotlight_mask
    img_array = np.clip(img_array, 0, 255)
    
    return Image.fromarray(img_array.astype(np.uint8))

def apply_wedding_ring_focus_v19(image: Image.Image) -> Image.Image:
    """Enhanced wedding ring processing - WITHOUT metallic highlight"""
    # Metallic highlight removed - skip this step entirely
    
    # 1. Center spotlight
    image = apply_center_spotlight_thumbnail(image, 0.08)
    
    # 2. Enhanced sharpness
    sharpness = ImageEnhance.Sharpness(image)
    image = sharpness.enhance(1.5)  # Increased from 1.3
    
    # 3. Contrast
    contrast = ImageEnhance.Contrast(image)
    image = contrast.enhance(1.08)  # Increased from 1.03
    
    # 4. Detail enhancement with stronger settings
    image = image.filter(ImageFilter.UnsharpMask(radius=1.5, percent=100, threshold=2))
    
    return image

def apply_basic_enhancement(image):
    """Apply basic enhancement - Increased brightness and contrast"""
    if image.mode != 'RGB':
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        else:
            image = image.convert('RGB')
    
    # Apply white balance correction
    image = auto_white_balance(image)
    
    # Basic enhancement (increased)
    brightness = ImageEnhance.Brightness(image)
    image = brightness.enhance(1.1)  # Increased to 1.1
    
    contrast = ImageEnhance.Contrast(image)
    image = contrast.enhance(1.08)  # Increased from 1.03
    
    color = ImageEnhance.Color(image)
    image = color.enhance(1.02)  # Slightly increased from 1.01
    
    return image

def apply_pattern_enhancement_v19(image, pattern_type, is_wedding_ring):
    """Apply enhancement based on pattern - Reduced brightness for wedding rings"""
    
    # Apply 3% white overlay to ALL patterns
    white_overlay = 0.03
    img_array = np.array(image, dtype=np.float32)
    img_array = img_array * (1 - white_overlay) + 255 * white_overlay
    img_array = np.clip(img_array, 0, 255)
    image = Image.fromarray(img_array.astype(np.uint8))
    
    if pattern_type in ["bc_only", "ac_only"]:
        # Unplated white
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.02)  # Increased from 0.98
        
        color = ImageEnhance.Color(image)
        image = color.enhance(0.95)
        
        # Additional white overlay (total 18%)
        white_overlay = 0.15
        img_array = np.array(image, dtype=np.float32)
        img_array = img_array * (1 - white_overlay) + 255 * white_overlay
        img_array = np.clip(img_array, 0, 255)
        image = Image.fromarray(img_array.astype(np.uint8))
        
    elif pattern_type in ["a_only", "b_only"]:
        # a_ and b_ patterns
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.08)  # Increased from 1.01
        
        color = ImageEnhance.Color(image)
        image = color.enhance(0.95)
        
        # Enhanced sharpness
        sharpness = ImageEnhance.Sharpness(image)
        image = sharpness.enhance(1.5)  # Increased from 1.25
        
    else:
        # Standard enhancement
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.05)  # Increased from 1.0
        
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.08)  # Increased from 1.02
    
    # Apply reduced center spotlight
    image = apply_center_spotlight_thumbnail(image, 0.10)  # Reduced from 0.15
    
    # Wedding ring special enhancement (always applied)
    image = apply_wedding_ring_focus_v19(image)
    
    return image

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
    buffered.seek(0)
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    return img_base64.rstrip('=')

def handler(event):
    """Thumbnail handler function - Wedding Ring Optimized"""
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
        
        # Get image - ULTRA SAFE extraction
        image = None
        
        if isinstance(input_data, dict):
            all_keys = list(input_data.keys())
            logger.info(f"Available keys: {all_keys[:20]}")
            
            # Priority order for image keys
            priority_keys = [
                'image_base64', 'imageBase64', 'enhanced_image', 'image',
                'base64', 'img', 'thumbnail', 'base64_image', 'raw_image'
            ]
            
            # Try priority keys first
            for key in priority_keys:
                if key in input_data and input_data[key]:
                    try:
                        value = input_data[key]
                        if isinstance(value, str) and len(value) > 50:
                            image = base64_to_image_ultra_safe(value)
                            logger.info(f"Successfully loaded image from priority key: {key}")
                            break
                    except Exception as e:
                        logger.warning(f"Failed to load from {key}: {str(e)}")
                        continue
            
            # If not found, try ALL keys
            if not image:
                for key in all_keys:
                    if key not in priority_keys and input_data[key]:
                        try:
                            value = input_data[key]
                            if isinstance(value, str) and len(value) > 50:
                                image = base64_to_image_ultra_safe(value)
                                logger.info(f"Successfully loaded image from key: {key}")
                                break
                        except:
                            continue
            
            # Try URL keys last
            if not image:
                url_keys = ['url', 'image_url', 'imageUrl', 'img_url']
                for key in url_keys:
                    if key in input_data and input_data[key]:
                        try:
                            image_url = input_data[key]
                            if image_url.startswith('http'):
                                image = download_image_from_url(image_url)
                                logger.info(f"Successfully loaded image from URL key: {key}")
                                break
                        except Exception as e:
                            logger.warning(f"Failed to load URL from {key}: {str(e)}")
                            continue
                            
        elif isinstance(input_data, str):
            try:
                if input_data.startswith('http'):
                    image = download_image_from_url(input_data)
                else:
                    image = base64_to_image_ultra_safe(input_data)
                logger.info("Successfully loaded image from direct string")
            except Exception as e:
                logger.error(f"Failed to load direct string: {str(e)}")
        
        if not image:
            logger.error(f"Failed to load image. Input type: {type(input_data)}")
            raise ValueError("Failed to load image from any source")
        
        # Apply basic enhancement
        enhanced_image = apply_basic_enhancement(image)
        
        # Always wedding ring
        is_wedding_ring = True
        logger.info(f"Wedding ring: Always True")
        
        # Apply Replicate enhancement if available
        replicate_applied = False
        if USE_REPLICATE:
            try:
                enhanced_image = apply_replicate_thumbnail_enhancement(enhanced_image, is_wedding_ring)
                replicate_applied = True
            except Exception as e:
                logger.error(f"Replicate thumbnail enhancement failed: {str(e)}")
                # Continue with basic enhancement instead of returning error
                logger.warning("Continuing with basic enhancement only")
        
        # Create thumbnail with upscaling
        thumbnail = create_thumbnail_smart_center_crop_with_upscale(enhanced_image, 1000, 1300)
        
        # Detect pattern type
        pattern_type = detect_pattern_type(filename)
        
        if pattern_type in ["bc_only", "ac_only"]:
            detected_type = "무도금화이트(0.15+0.03)"
        elif pattern_type in ["b_only", "a_only"]:
            detected_type = f"{pattern_type[0]}_패턴(0.03)"
        else:
            detected_type = "기타색상(0.03)"
        
        # Apply pattern-specific enhancement
        thumbnail = apply_pattern_enhancement_v19(thumbnail, pattern_type, is_wedding_ring)
        
        # Final sharpness (increased for wedding rings)
        sharpness = ImageEnhance.Sharpness(thumbnail)
        thumbnail = sharpness.enhance(1.6)  # Increased from 1.3
        
        # Final brightness touch (increased)
        brightness = ImageEnhance.Brightness(thumbnail)
        thumbnail = brightness.enhance(1.05)  # Increased from 1.02
        
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
                "is_wedding_ring": True,
                "filename": output_filename,
                "original_filename": filename,
                "image_index": image_index,
                "format": "base64_no_padding",
                "has_spotlight": True,
                "has_upscaling": True,
                "version": VERSION,
                "status": "success",
                "replicate_applied": replicate_applied,
                "white_overlay_applied": "3% base + pattern specific",
                "center_spotlight": "10% reduced",
                "base64_decode_method": "ultra_safe_v19",
                "wedding_ring_enhancement": "cubic_focus_only_no_metallic"
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
