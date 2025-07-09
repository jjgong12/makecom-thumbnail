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
import cv2

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

################################
# THUMBNAIL HANDLER - 1000x1300
# VERSION: V5.4-Deeper-Gray
################################

VERSION = "V5.4-Deeper-Gray"

# ===== REPLICATE INITIALIZATION =====
REPLICATE_API_TOKEN = os.environ.get('REPLICATE_API_TOKEN')
REPLICATE_CLIENT = None
USE_REPLICATE = False

if REPLICATE_API_TOKEN:
    try:
        REPLICATE_CLIENT = replicate.Client(api_token=REPLICATE_API_TOKEN)
        USE_REPLICATE = True
        logger.info("✅ Replicate client initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Replicate: {e}")
        USE_REPLICATE = False

def find_input_data_fast(data):
    """Fast input data extraction"""
    if isinstance(data, str) and len(data) > 50:
        return {'image': data}
    
    if isinstance(data, dict):
        # Priority keys
        priority_keys = ['image', 'image_base64', 'enhanced_image', 'base64', 'img']
        
        for key in priority_keys:
            if key in data and isinstance(data[key], str) and len(data[key]) > 50:
                return {key: data[key]}
        
        # Check nested (limited depth)
        for key in ['input', 'data']:
            if key in data and isinstance(data[key], dict):
                result = find_input_data_fast(data[key])
                if result:
                    return result
        
        # Numeric keys (Make.com)
        for i in range(5):
            if str(i) in data and isinstance(data[str(i)], str) and len(data[str(i)]) > 50:
                return {'image': data[str(i)]}
    
    return data

def find_filename_fast(data):
    """Fast filename extraction"""
    if isinstance(data, dict):
        for key in ['filename', 'file_name', 'name']:
            if key in data and isinstance(data[key], str):
                return data[key]
        
        # Check nested once
        if 'input' in data and isinstance(data['input'], dict):
            for key in ['filename', 'file_name', 'name']:
                if key in data['input']:
                    return data['input'][key]
    
    return None

def generate_thumbnail_filename(original_filename, image_index):
    """Generate thumbnail filename"""
    if not original_filename:
        return f"thumbnail_{image_index:03d}.jpg"
    
    thumbnail_numbers = {1: "007", 3: "008", 5: "009"}
    
    new_filename = original_filename
    pattern = r'(_\d{3})'
    if re.search(pattern, new_filename):
        new_filename = re.sub(pattern, f'_{thumbnail_numbers.get(image_index, "007")}', new_filename)
    else:
        name_parts = new_filename.split('.')
        name_parts[0] += f'_{thumbnail_numbers.get(image_index, "007")}'
        new_filename = '.'.join(name_parts)
    
    return new_filename

def base64_to_image_fast(base64_string):
    """Fast base64 to image conversion - PNG support"""
    try:
        if not base64_string or len(base64_string) < 50:
            raise ValueError("Invalid base64")
        
        # Quick clean
        if 'base64,' in base64_string:
            base64_string = base64_string.split('base64,')[-1]
        
        # Remove whitespace
        base64_string = ''.join(base64_string.split())
        
        # Keep only valid chars
        valid_chars = set(string.ascii_letters + string.digits + '+/=')
        base64_string = ''.join(c for c in base64_string if c in valid_chars)
        
        # Try without padding first (Make.com)
        no_pad = base64_string.rstrip('=')
        
        try:
            img_data = base64.b64decode(no_pad, validate=False)
            return Image.open(BytesIO(img_data))
        except:
            # Try with correct padding
            padding = (4 - len(no_pad) % 4) % 4
            padded = no_pad + ('=' * padding)
            img_data = base64.b64decode(padded, validate=False)
            return Image.open(BytesIO(img_data))
            
    except Exception as e:
        logger.error(f"Base64 decode error: {str(e)}")
        raise ValueError(f"Invalid base64: {str(e)}")

def detect_pattern_type(filename: str) -> str:
    """Detect pattern type - SIMPLIFIED to ac_ and others"""
    if not filename:
        return "other"
    
    filename_lower = filename.lower()
    
    # Only ac_ is special (무도금화이트)
    if 'ac_' in filename_lower:
        return "ac_pattern"
    else:
        return "other"

def create_background(size, color="#CCCCCC", style="gradient"):
    """Create natural gray background for jewelry - V5.4 DEEPER GRAY"""
    width, height = size
    
    if style == "gradient":
        # Create radial gradient background
        background = Image.new('RGB', size, color)
        bg_array = np.array(background, dtype=np.float32)
        
        # Create radial gradient
        y, x = np.ogrid[:height, :width]
        center_x, center_y = width / 2, height / 2
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2) / max(width, height)
        
        # Subtle gradient for natural look with slightly darker edges
        gradient = 1 - (distance * 0.10)  # 10% darkening at edges (increased from 8%)
        gradient = np.clip(gradient, 0.90, 1.0)
        
        # Apply gradient
        bg_array *= gradient[:, :, np.newaxis]
        
        return Image.fromarray(bg_array.astype(np.uint8))
    else:
        return Image.new('RGB', size, color)

def remove_background_with_replicate(image: Image.Image) -> Image.Image:
    """Remove background using Replicate API - V5.4 MORE AGGRESSIVE"""
    if not USE_REPLICATE or not REPLICATE_CLIENT:
        logger.warning("Replicate not available for background removal")
        return image
    
    try:
        logger.info("🔷 Removing background with Replicate (V5.4 extra aggressive)")
        
        # Convert to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        buffered.seek(0)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        img_data_url = f"data:image/png;base64,{img_base64}"
        
        # Use rembg model with EVEN MORE AGGRESSIVE settings
        output = REPLICATE_CLIENT.run(
            "cjwbw/rembg:fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003",
            input={
                "image": img_data_url,
                "model": "u2net",  # u2net for better hole detection
                "alpha_matting": True,
                "alpha_matting_foreground_threshold": 280,  # Even higher for more aggressive
                "alpha_matting_background_threshold": 5,     # Even lower for holes
                "alpha_matting_erode_size": 12              # Larger for cleaner edges
            }
        )
        
        if output:
            if isinstance(output, str):
                response = requests.get(output)
                result_image = Image.open(BytesIO(response.content))
            else:
                result_image = Image.open(BytesIO(base64.b64decode(output)))
            
            # Additional check for ring holes
            if result_image.mode == 'RGBA':
                result_image = ensure_ring_holes_transparent(result_image)
            
            logger.info("✅ Background removal successful")
            return result_image
        else:
            logger.warning("No output from background removal")
            return image
            
    except Exception as e:
        logger.error(f"Background removal error: {str(e)}")
        return image

def ensure_ring_holes_transparent(image: Image.Image) -> Image.Image:
    """Ensure ring holes are properly transparent - V5.4 NEW"""
    if image.mode != 'RGBA':
        return image
    
    # Get alpha channel
    r, g, b, a = image.split()
    alpha_array = np.array(a)
    
    # Find potential ring hole areas (closed regions with low alpha)
    # Use flood fill to detect enclosed areas
    h, w = alpha_array.shape
    
    # Create a padded version for flood fill
    padded = np.pad(alpha_array, 1, mode='constant', constant_values=0)
    
    # Flood fill from outside
    filled = padded.copy()
    cv2.floodFill(filled, None, (0, 0), 255)
    
    # Remove padding
    filled = filled[1:-1, 1:-1]
    
    # Holes are areas that weren't filled
    holes = (filled != 255) & (alpha_array < 200)
    
    # Set hole areas to fully transparent
    alpha_array[holes] = 0
    
    # Apply slight erosion to clean edges around holes
    kernel = np.ones((3, 3), np.uint8)
    alpha_array = cv2.morphologyEx(alpha_array, cv2.MORPH_OPEN, kernel)
    
    # Create new image with corrected alpha
    a_new = Image.fromarray(alpha_array)
    return Image.merge('RGBA', (r, g, b, a_new))

def add_natural_edge_feathering(image: Image.Image) -> Image.Image:
    """Add MINIMAL natural feathering to edges - V5.4 MINIMAL"""
    if image.mode != 'RGBA':
        return image
    
    # Get alpha channel
    r, g, b, a = image.split()
    
    # Apply minimal Gaussian blur to alpha channel for natural edges
    a_array = np.array(a, dtype=np.float32)
    
    # Just apply a simple, subtle blur to the alpha channel
    # No edge detection, no complex processing - just natural softening
    alpha_blurred = cv2.GaussianBlur(a_array, (3, 3), 1.0)  # Very subtle blur
    
    # Create new image with slightly blurred alpha
    a_new = Image.fromarray(alpha_blurred.astype(np.uint8))
    return Image.merge('RGBA', (r, g, b, a_new))

def composite_with_light_gray_background(image, background_color="#CCCCCC"):
    """Natural composite WITHOUT shadow - V5.4 DEEPER GRAY"""
    if image.mode == 'RGBA':
        # Apply minimal edge feathering first
        image = add_natural_edge_feathering(image)
        
        # Create background
        background = create_background(image.size, background_color, style="gradient")
        
        # NO SHADOW - Direct composite only
        # Use simple alpha blending without any shadow effects
        r, g, b, a = image.split()
        
        # Convert to arrays for blending
        fg_array = np.array(image.convert('RGB'), dtype=np.float32)
        bg_array = np.array(background, dtype=np.float32)
        alpha_array = np.array(a, dtype=np.float32) / 255.0
        
        # Simple alpha blending - no premultiplication, no complex processing
        # This creates the most natural transition
        for i in range(3):
            bg_array[:,:,i] = fg_array[:,:,i] * alpha_array + bg_array[:,:,i] * (1 - alpha_array)
        
        # Convert back
        result = Image.fromarray(bg_array.astype(np.uint8))
        return result
    else:
        return image

def apply_swinir_thumbnail_after_resize(image: Image.Image) -> Image.Image:
    """Apply SwinIR AFTER resize - NEW for thumbnails"""
    if not USE_REPLICATE or not REPLICATE_CLIENT:
        return image
    
    try:
        width, height = image.size
        
        # Only apply to thumbnail size
        if width > 1200 or height > 1500:
            logger.info(f"Skipping SwinIR - too large: {width}x{height}")
            return image
        
        logger.info(f"Applying SwinIR to thumbnail: {width}x{height}")
        
        # Convert to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG", optimize=False)
        buffered.seek(0)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        img_data_url = f"data:image/png;base64,{img_base64}"
        
        logger.info("🔷 SwinIR thumbnail (post-resize)")
        
        output = REPLICATE_CLIENT.run(
            "jingyunliang/swinir:660d922d33153019e8c263a3bba265de882e7f4f70396546b6c9c8f9d47a021a",
            input={
                "image": img_data_url,
                "task_type": "Real-World Image Super-Resolution",
                "scale": 1,  # Keep size
                "noise_level": 10,
                "jpeg_quality": 50
            }
        )
        
        if output:
            if isinstance(output, str):
                response = requests.get(output)
                enhanced_image = Image.open(BytesIO(response.content))
            else:
                enhanced_image = Image.open(BytesIO(base64.b64decode(output)))
            
            return enhanced_image
            
    except Exception as e:
        logger.warning(f"SwinIR error: {str(e)}")
        
    return image

def enhance_cubic_details_thumbnail_simple(image: Image.Image) -> Image.Image:
    """Enhanced cubic details for thumbnails - V5.4"""
    # Moderate contrast
    contrast = ImageEnhance.Contrast(image)
    image = contrast.enhance(1.10)  # Moderate contrast
    
    # Moderate detail enhancement
    image = image.filter(ImageFilter.UnsharpMask(radius=0.4, percent=130, threshold=2))  # Moderate enhancement
    
    # Subtle micro-contrast
    contrast2 = ImageEnhance.Contrast(image)
    image = contrast2.enhance(1.04)  # Subtle contrast
    
    # Moderate sharpness
    sharpness = ImageEnhance.Sharpness(image)
    image = sharpness.enhance(1.25)  # Moderate sharpness
    
    return image

def auto_white_balance_fast(image: Image.Image) -> Image.Image:
    """Fast white balance"""
    img_array = np.array(image, dtype=np.float32)
    
    # Sample for speed
    sampled = img_array[::10, ::10]
    gray_mask = (
        (np.abs(sampled[:,:,0] - sampled[:,:,1]) < 15) & 
        (np.abs(sampled[:,:,1] - sampled[:,:,2]) < 15) &
        (sampled[:,:,0] > 180)
    )
    
    if np.sum(gray_mask) > 10:
        r_avg = np.mean(sampled[gray_mask, 0])
        g_avg = np.mean(sampled[gray_mask, 1])
        b_avg = np.mean(sampled[gray_mask, 2])
        
        gray_avg = (r_avg + g_avg + b_avg) / 3
        
        img_array[:,:,0] *= (gray_avg / r_avg) if r_avg > 0 else 1
        img_array[:,:,1] *= (gray_avg / g_avg) if g_avg > 0 else 1
        img_array[:,:,2] *= (gray_avg / b_avg) if b_avg > 0 else 1
    
    return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

def apply_center_spotlight_fast(image: Image.Image, intensity: float = 0.035) -> Image.Image:
    """Fast center spotlight - V5.4"""
    width, height = image.size
    
    y, x = np.ogrid[:height, :width]
    center_x, center_y = width / 2, height / 2
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2) / max(width, height)
    
    spotlight_mask = 1 + intensity * np.exp(-distance**2 * 3)
    spotlight_mask = np.clip(spotlight_mask, 1.0, 1.0 + intensity)
    
    img_array = np.array(image, dtype=np.float32)
    img_array *= spotlight_mask[:, :, np.newaxis]
    
    return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

def apply_wedding_ring_focus_fast(image: Image.Image) -> Image.Image:
    """Enhanced wedding ring focus for thumbnails - V5.4"""
    # Moderate spotlight
    image = apply_center_spotlight_fast(image, 0.025)  # Subtle spotlight
    
    # Moderate sharpness
    sharpness = ImageEnhance.Sharpness(image)
    image = sharpness.enhance(1.8)  # Moderate sharpness
    
    # Moderate contrast
    contrast = ImageEnhance.Contrast(image)
    image = contrast.enhance(1.05)  # Subtle contrast
    
    # Moderate multi-scale unsharp mask
    image = image.filter(ImageFilter.UnsharpMask(radius=1.0, percent=110, threshold=2))  # Moderate enhancement
    
    return image

def calculate_quality_metrics_fast(image: Image.Image) -> dict:
    """Fast quality metrics"""
    img_array = np.array(image)[::20, ::20]  # Sample
    
    r_avg = np.mean(img_array[:,:,0])
    g_avg = np.mean(img_array[:,:,1])
    b_avg = np.mean(img_array[:,:,2])
    
    brightness = (r_avg + g_avg + b_avg) / 3
    cool_tone_diff = b_avg - r_avg
    
    return {
        "brightness": brightness,
        "cool_tone_diff": cool_tone_diff
    }

def apply_pattern_enhancement_fast(image, pattern_type):
    """Fast pattern enhancement - 15% white overlay for ac_ (1차) - V5.4"""
    
    # Apply white overlay ONLY to ac_pattern (15% - reduced from 17%)
    if pattern_type == "ac_pattern":
        # Unplated white - 15% white overlay - V5.4
        white_overlay = 0.15
        img_array = np.array(image, dtype=np.float32)
        img_array = img_array * (1 - white_overlay) + 255 * white_overlay
        img_array = np.clip(img_array, 0, 255)
        image = Image.fromarray(img_array.astype(np.uint8))
        
        # Reduced brightness for ac_
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.01)  # Subtle brightness
        
        color = ImageEnhance.Color(image)
        image = color.enhance(0.97)  # Slightly desaturated
        
    else:
        # All other patterns - moderate enhancement
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.10)  # Moderate brightness
        
        color = ImageEnhance.Color(image)
        image = color.enhance(0.98)
        
        # Moderate sharpness
        sharpness = ImageEnhance.Sharpness(image)
        image = sharpness.enhance(1.7)  # Moderate sharpness
    
    # Moderate spotlight
    image = apply_center_spotlight_fast(image, 0.035)  # Moderate intensity
    
    # Wedding ring enhancement
    image = apply_wedding_ring_focus_fast(image)
    
    # Fast quality check (only for ac_pattern) - 2차 처리
    if pattern_type == "ac_pattern":
        metrics = calculate_quality_metrics_fast(image)
        if metrics["brightness"] < 240:
            # Apply 18% white overlay as correction - V5.4
            white_overlay = 0.18  # Secondary correction
            img_array = np.array(image, dtype=np.float32)
            img_array = img_array * (1 - white_overlay) + 255 * white_overlay
            img_array = np.clip(img_array, 0, 255)
            image = Image.fromarray(img_array.astype(np.uint8))
    
    return image

def create_thumbnail_optimized(image, target_width=1000, target_height=1300):
    """Optimized thumbnail creation for 2000x2600 input"""
    original_width, original_height = image.size
    
    # Check if input is expected 2000x2600 ratio
    expected_ratio = 2000 / 2600  # 0.769
    actual_ratio = original_width / original_height
    
    if abs(actual_ratio - expected_ratio) < 0.01:
        # Perfect ratio match - direct resize to 1000x1300
        return image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    else:
        # Different ratio - use existing logic
        logger.warning(f"Unexpected ratio: {original_width}x{original_height} ({actual_ratio:.3f})")
        
        # Fixed center
        image_center = (original_width // 2, original_height // 2)
        
        # Upscale if needed
        if original_width < target_width or original_height < target_height:
            scale_factor = max(target_width / original_width, target_height / original_height) * 1.1
            
            new_size = (int(original_width * scale_factor), int(original_height * scale_factor))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            image_center = (new_size[0] // 2, new_size[1] // 2)
            original_width, original_height = new_size
        
        # Check for specific sizes
        if ((1800 <= original_width <= 2200 and 2400 <= original_height <= 2800) or
            (2800 <= original_width <= 3200 and 3700 <= original_height <= 4100)):
            
            crop_ratio = 0.75 if original_width >= 2800 else 0.85
            
            crop_width = int(original_width * crop_ratio)
            crop_height = int(original_height * crop_ratio)
            
            left = image_center[0] - crop_width // 2
            top = image_center[1] - crop_height // 2
            
            cropped = image.crop((left, top, left + crop_width, top + crop_height))
            return cropped.resize((target_width, target_height), Image.Resampling.LANCZOS)
        else:
            # Simple resize
            width_ratio = target_width / original_width
            height_ratio = target_height / original_height
            scale_ratio = min(width_ratio, height_ratio)
            
            new_size = (int(original_width * scale_ratio), int(original_height * scale_ratio))
            resized = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Center on white background
            thumbnail = Image.new('RGB', (target_width, target_height), (255, 255, 255))
            left = (target_width - new_size[0]) // 2
            top = (target_height - new_size[1]) // 2
            thumbnail.paste(resized, (left, top))
            
            return thumbnail

def image_to_base64(image):
    """Convert to base64 without padding"""
    buffered = BytesIO()
    
    if image.mode == 'RGBA':
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
    
    image.save(buffered, format='PNG', optimize=False, quality=95)
    buffered.seek(0)
    
    return base64.b64encode(buffered.getvalue()).decode().rstrip('=')

def handler(event):
    """Optimized thumbnail handler - V5.4 DEEPER GRAY VERSION"""
    try:
        logger.info(f"=== Thumbnail {VERSION} Started ===")
        
        # Get image index
        image_index = event.get('image_index', 1)
        if isinstance(event.get('input'), dict):
            image_index = event.get('input', {}).get('image_index', image_index)
        
        # Fixed gray background - DEEPER V5.4
        background_color = '#CCCCCC'  # Deeper gray background
        
        # Fast extraction
        filename = find_filename_fast(event)
        input_data = find_input_data_fast(event)
        
        if not input_data:
            raise ValueError("No input data found")
        
        # Get image
        image = None
        priority_keys = ['image', 'image_base64', 'enhanced_image', 'base64']
        
        for key in priority_keys:
            if key in input_data and input_data[key]:
                try:
                    image = base64_to_image_fast(input_data[key])
                    break
                except:
                    continue
        
        if not image:
            raise ValueError("Failed to load image")
        
        # STEP 1: BACKGROUND REMOVAL (PNG only)
        original_mode = image.mode
        has_transparency = image.mode == 'RGBA'
        needs_background_removal = False
        
        if filename and filename.lower().endswith('.png'):
            logger.info("📸 STEP 1: PNG detected - removing background with V5.4 settings")
            image = remove_background_with_replicate(image)
            has_transparency = image.mode == 'RGBA'
            needs_background_removal = True
        
        # Keep transparent version for later
        if has_transparency:
            original_transparent = image.copy()
        
        # Convert to RGB for processing
        if image.mode != 'RGB':
            if image.mode == 'RGBA':
                temp_bg = Image.new('RGB', image.size, (255, 255, 255))
                temp_bg.paste(image, mask=image.split()[3])
                image = temp_bg
            else:
                image = image.convert('RGB')
        
        # STEP 2: ENHANCEMENT
        logger.info("🎨 STEP 2: Applying enhancements")
        
        # Fast white balance
        image = auto_white_balance_fast(image)
        
        # Moderate basic enhancement - V5.4
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.12)  # Moderate brightness
        
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.06)  # Moderate contrast
        
        color = ImageEnhance.Color(image)
        image = color.enhance(1.01)
        
        # Detect pattern
        pattern_type = detect_pattern_type(filename)
        
        # Create thumbnail
        thumbnail = create_thumbnail_optimized(image, 1000, 1300)
        
        # Apply SwinIR AFTER resize
        swinir_applied = False
        if USE_REPLICATE:
            try:
                logger.info("Applying SwinIR enhancement")
                thumbnail = apply_swinir_thumbnail_after_resize(thumbnail)
                swinir_applied = True
            except:
                logger.warning("SwinIR failed, continuing without")
        
        # Enhanced cubic details
        thumbnail = enhance_cubic_details_thumbnail_simple(thumbnail)
        
        detected_type = {
            "ac_pattern": "무도금화이트(0.15/0.18)",
            "other": "기타색상(no_overlay)"
        }.get(pattern_type, "기타색상")
        
        # Apply pattern enhancement (includes 2차 처리)
        thumbnail = apply_pattern_enhancement_fast(thumbnail, pattern_type)
        
        # STEP 3: BACKGROUND COMPOSITE (if transparent)
        if has_transparency and 'original_transparent' in locals():
            logger.info(f"🖼️ STEP 3: Natural background compositing (NO SHADOW): {background_color}")
            
            # Apply enhancements to transparent version
            enhanced_transparent = original_transparent.resize((1000, 1300), Image.Resampling.LANCZOS)
            
            if enhanced_transparent.mode == 'RGBA':
                # Ensure ring holes are transparent
                enhanced_transparent = ensure_ring_holes_transparent(enhanced_transparent)
                
                # Split channels
                r, g, b, a = enhanced_transparent.split()
                rgb_image = Image.merge('RGB', (r, g, b))
                
                # Apply same enhancements
                rgb_image = auto_white_balance_fast(rgb_image)
                rgb_image = enhance_cubic_details_thumbnail_simple(rgb_image)
                brightness = ImageEnhance.Brightness(rgb_image)
                rgb_image = brightness.enhance(1.12)  # Moderate brightness
                contrast = ImageEnhance.Contrast(rgb_image)
                rgb_image = contrast.enhance(1.06)  # Moderate contrast
                sharpness = ImageEnhance.Sharpness(rgb_image)
                rgb_image = sharpness.enhance(1.8)  # Moderate sharpness
                
                # Pattern enhancement based on type
                if pattern_type == "ac_pattern":
                    # 15% white overlay - V5.4
                    white_overlay = 0.15
                    img_array = np.array(rgb_image, dtype=np.float32)
                    img_array = img_array * (1 - white_overlay) + 255 * white_overlay
                    rgb_image = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
                
                # Merge back
                r2, g2, b2 = rgb_image.split()
                enhanced_transparent = Image.merge('RGBA', (r2, g2, b2, a))
            
            # Natural composite with deeper gray background
            thumbnail = composite_with_light_gray_background(enhanced_transparent, background_color)
            
            # Final sharpness after compositing
            sharpness = ImageEnhance.Sharpness(thumbnail)
            thumbnail = sharpness.enhance(1.15)  # Subtle sharpness for natural look
        
        # Final adjustments - V5.4
        sharpness = ImageEnhance.Sharpness(thumbnail)
        thumbnail = sharpness.enhance(1.7)  # Moderate sharpness
        
        brightness = ImageEnhance.Brightness(thumbnail)
        thumbnail = brightness.enhance(1.03)  # Slight brightness boost
        
        # Convert to base64
        thumbnail_base64 = image_to_base64(thumbnail)
        
        # Generate filename
        output_filename = generate_thumbnail_filename(filename, image_index)
        
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
                "version": VERSION,
                "status": "success",
                "swinir_applied": swinir_applied,
                "swinir_timing": "AFTER resize",
                "png_support": True,
                "has_transparency": has_transparency,
                "background_composite": has_transparency,
                "background_removal": needs_background_removal,
                "background_color": background_color,
                "background_style": "Deeper gray gradient (#CCCCCC)",
                "gradient_edge_darkening": "10%",
                "shadow": "REMOVED - No shadow for natural look",
                "edge_processing": "Minimal natural feathering (3x3 blur)",
                "composite_method": "Simple alpha blending",
                "rembg_settings": "Extra aggressive (280/5/12)",
                "ring_hole_detection": "Flood fill detection + morphology",
                "expected_input": "2000x2600",
                "output_size": "1000x1300",
                "cubic_enhancement": "Moderate (130% unsharp)",
                "white_overlay": "15% for ac_ (1차), 18% (2차)",
                "brightness_increased": "12%",
                "contrast_increased": "6%", 
                "sharpness_increased": "1.7-1.8 + extra passes",
                "spotlight_increased": "2.5-3.5%",
                "processing_order": "1.Background Removal → 2.Enhancement → 3.Natural Composite (No Shadow)",
                "quality": "95"
            }
        }
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
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
