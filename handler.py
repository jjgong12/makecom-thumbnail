import runpod
import os
import base64
import numpy as np
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter
import logging
import re
import string
import cv2

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

VERSION = "V30-Advanced-Detail-Enhancement"

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
    """Fast base64 to image conversion"""
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
    """Detect pattern type"""
    if not filename:
        return "other"
    
    filename_lower = filename.lower()
    
    if 'bc_' in filename_lower or 'ac_' in filename_lower:
        return "bc_ac"
    elif 'b_' in filename_lower and 'bc_' not in filename_lower:
        return "b_only"
    elif 'a_' in filename_lower and 'ac_' not in filename_lower:
        return "a_only"
    else:
        return "other"

def enhance_cubic_details_thumbnail_advanced(image: Image.Image) -> Image.Image:
    """Advanced cubic enhancement for thumbnails"""
    
    # 1. Multi-scale unsharp mask optimized for thumbnails
    # Smaller radii for thumbnail size
    large_detail = image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=130, threshold=2))
    medium_detail = large_detail.filter(ImageFilter.UnsharpMask(radius=0.6, percent=110, threshold=1))
    fine_detail = medium_detail.filter(ImageFilter.UnsharpMask(radius=0.3, percent=100, threshold=1))
    
    # 2. Edge enhancement for sparkle
    edges = fine_detail.filter(ImageFilter.EDGE_ENHANCE)
    
    # 3. Blend edge enhancement
    enhanced = Image.blend(fine_detail, edges, 0.25)
    
    # 4. Micro-contrast for thumbnail visibility
    contrast = ImageEnhance.Contrast(enhanced)
    enhanced = contrast.enhance(1.08)
    
    # 5. Detail filter
    enhanced = enhanced.filter(ImageFilter.DETAIL)
    
    return enhanced

def enhance_jewelry_details_thumbnail(image: Image.Image, pattern_type: str) -> Image.Image:
    """Jewelry-specific detail enhancement for thumbnails"""
    try:
        img_array = np.array(image)
        
        # 1. Simplified CLAHE for thumbnails
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Less aggressive CLAHE for thumbnails
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4,4))
        l = clahe.apply(l)
        
        lab = cv2.merge([l, a, b])
        img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # 2. Selective sharpening for bright areas
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Find bright areas (cubics)
        _, bright_mask = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)
        
        # Smaller kernel for thumbnails
        kernel = np.ones((2,2), np.uint8)
        bright_mask = cv2.dilate(bright_mask, kernel, iterations=1)
        
        # Gentler sharpening kernel for thumbnails
        sharpen_kernel = np.array([[0,-1,0],
                                  [-1,5,-1],
                                  [0,-1,0]])
        
        sharpened = cv2.filter2D(img_array, -1, sharpen_kernel)
        
        # Blend based on mask
        mask_3d = np.stack([bright_mask/255]*3, axis=2).astype(np.float32)
        result = img_array * (1 - mask_3d * 0.4) + sharpened * (mask_3d * 0.4)
        
        # 3. Edge enhancement for white/unplated patterns
        if pattern_type in ["bc_ac", "b_only"]:
            # Subtle edge enhancement
            edges = cv2.Canny(gray, 50, 150)
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            result = cv2.addWeighted(result.astype(np.uint8), 0.95, edges_colored, 0.05, 0)
        
        return Image.fromarray(result.astype(np.uint8))
        
    except Exception as e:
        logger.warning(f"OpenCV enhancement failed: {e}, using PIL fallback")
        return enhance_cubic_details_thumbnail_advanced(image)

def enhance_cubic_details_thumbnail_fast_quality(image: Image.Image, pattern_type: str) -> Image.Image:
    """Main thumbnail detail enhancement function"""
    
    # 1. Initial sharpening for thumbnails
    sharpness = ImageEnhance.Sharpness(image)
    image = sharpness.enhance(1.25)
    
    # 2. Apply advanced multi-scale enhancement
    image = enhance_cubic_details_thumbnail_advanced(image)
    
    # 3. Apply jewelry-specific enhancement
    image = enhance_jewelry_details_thumbnail(image, pattern_type)
    
    # 4. Pattern-specific optimization for thumbnails
    if pattern_type in ["bc_ac", "b_only"]:
        # Extra contrast for white/unplated patterns
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.04)
        
        # Fine detail boost
        image = image.filter(ImageFilter.UnsharpMask(radius=0.2, percent=70, threshold=1))
    
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

def apply_center_spotlight_fast(image: Image.Image, intensity: float = 0.03) -> Image.Image:
    """Fast center spotlight"""
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
    """Fast wedding ring enhancement with cubic focus"""
    # Reduced spotlight
    image = apply_center_spotlight_fast(image, 0.02)
    
    # Enhanced sharpness for cubic visibility
    sharpness = ImageEnhance.Sharpness(image)
    image = sharpness.enhance(1.6)  # Increased for thumbnails
    
    # Slight contrast
    contrast = ImageEnhance.Contrast(image)
    image = contrast.enhance(1.03)
    
    # Multi-scale unsharp mask
    image = image.filter(ImageFilter.UnsharpMask(radius=0.8, percent=90, threshold=2))
    
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
    """Fast pattern enhancement - Modified white overlay (10% primary, 3% additional)"""
    
    # Apply white overlay ONLY to bc_ac pattern (10% primary)
    if pattern_type == "bc_ac":
        # Unplated white - 10% white overlay
        white_overlay = 0.10
        img_array = np.array(image, dtype=np.float32)
        img_array = img_array * (1 - white_overlay) + 255 * white_overlay
        img_array = np.clip(img_array, 0, 255)
        image = Image.fromarray(img_array.astype(np.uint8))
        
        # Reduced brightness
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.02)  # Reduced from 1.05
        
        color = ImageEnhance.Color(image)
        image = color.enhance(0.96)
        
    elif pattern_type in ["a_only", "b_only"]:
        # NO white overlay
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.06)  # Reduced from 1.10
        
        color = ImageEnhance.Color(image)
        image = color.enhance(0.96)
        
        # Increased sharpness for gold patterns
        sharpness = ImageEnhance.Sharpness(image)
        image = sharpness.enhance(1.5)
        
    else:
        # Standard - NO white overlay
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.04)  # Reduced from 1.08
        
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.02)
    
    # Reduced spotlight
    if pattern_type in ["a_only", "b_only"]:
        image = apply_center_spotlight_fast(image, 0.02)
    else:
        image = apply_center_spotlight_fast(image, 0.03)
    
    # Wedding ring enhancement
    image = apply_wedding_ring_focus_fast(image)
    
    # Fast quality check (only for bc_ac)
    if pattern_type == "bc_ac":
        metrics = calculate_quality_metrics_fast(image)
        if metrics["brightness"] < 240:
            # Apply 3% additional white overlay
            white_overlay = 0.03  # Total 13%
            img_array = np.array(image, dtype=np.float32)
            img_array = img_array * (1 - white_overlay) + 255 * white_overlay
            img_array = np.clip(img_array, 0, 255)
            image = Image.fromarray(img_array.astype(np.uint8))
    
    return image

def create_thumbnail_fast(image, target_width=1000, target_height=1300):
    """Fast thumbnail creation"""
    original_width, original_height = image.size
    
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
    """Optimized thumbnail handler - ADVANCED DETAIL ENHANCEMENT"""
    try:
        logger.info(f"=== Thumbnail {VERSION} Started ===")
        
        # Get image index
        image_index = event.get('image_index', 1)
        if isinstance(event.get('input'), dict):
            image_index = event.get('input', {}).get('image_index', image_index)
        
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
        
        # Basic enhancement
        if image.mode != 'RGB':
            if image.mode == 'RGBA':
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])
                image = background
            else:
                image = image.convert('RGB')
        
        # Fast white balance
        image = auto_white_balance_fast(image)
        
        # Basic enhancement (reduced)
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.08)  # Reduced from 1.12
        
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.02)
        
        color = ImageEnhance.Color(image)
        image = color.enhance(1.01)
        
        # Detect pattern
        pattern_type = detect_pattern_type(filename)
        
        # Apply advanced detail enhancement (replaces SwinIR)
        logger.info("Applying advanced detail enhancement for thumbnail")
        image = enhance_cubic_details_thumbnail_fast_quality(image, pattern_type)
        
        # Create thumbnail
        thumbnail = create_thumbnail_fast(image, 1000, 1300)
        
        detected_type = {
            "bc_ac": "무도금화이트(0.10+0.03)",
            "b_only": "b_패턴(no_overlay+spotlight2%)",
            "a_only": "a_패턴(no_overlay+spotlight2%)",
            "other": "기타색상(no_overlay)"
        }.get(pattern_type, "기타색상")
        
        # Apply pattern enhancement
        thumbnail = apply_pattern_enhancement_fast(thumbnail, pattern_type)
        
        # Final adjustments
        sharpness = ImageEnhance.Sharpness(thumbnail)
        thumbnail = sharpness.enhance(1.7)  # Increased for cubic clarity
        
        brightness = ImageEnhance.Brightness(thumbnail)
        thumbnail = brightness.enhance(1.03)  # Reduced from 1.06
        
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
                "detail_enhancement": "Advanced multi-scale + OpenCV CLAHE",
                "white_overlay": "10% primary + 3% additional for bc_ac, 0% others",
                "brightness_reduced": True,
                "sharpness_increased": "1.6-1.7 + multi-scale",
                "spotlight_reduced": "2-3%",
                "enhancements_applied": [
                    "Multi-scale unsharp mask (3 levels)",
                    "OpenCV CLAHE for local contrast",
                    "Selective sharpening for bright areas",
                    "Edge enhancement for sparkle",
                    "Pattern-specific optimization"
                ],
                "processing_order": "White Balance → Advanced Detail Enhancement → Thumbnail Creation → Pattern Enhancement"
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
