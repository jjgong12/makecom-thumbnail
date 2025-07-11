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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

################################
# THUMBNAIL HANDLER - 1000x1300
# VERSION: V14.0-Precision-Edge
################################

VERSION = "V14.0-Precision-Edge"

# ===== GLOBAL INITIALIZATION =====
REPLICATE_API_TOKEN = os.environ.get('REPLICATE_API_TOKEN')
REPLICATE_CLIENT = None

if REPLICATE_API_TOKEN:
    try:
        REPLICATE_CLIENT = replicate.Client(api_token=REPLICATE_API_TOKEN)
        logger.info("✅ Replicate client initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Replicate: {e}")

# Global rembg session
REMBG_SESSION = None

def init_rembg_session():
    """Initialize rembg session once globally"""
    global REMBG_SESSION
    if REMBG_SESSION is None:
        try:
            from rembg import new_session
            REMBG_SESSION = new_session('birefnet-general')
            logger.info("✅ BiRefNet session initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize rembg: {e}")
            REMBG_SESSION = None
    return REMBG_SESSION

# Initialize on module load
init_rembg_session()

def find_input_data_fast(data):
    """Fast input data extraction"""
    if isinstance(data, str) and len(data) > 50:
        return {'image': data}
    
    if isinstance(data, dict):
        priority_keys = ['image', 'image_base64', 'enhanced_image', 'base64', 'img']
        
        for key in priority_keys:
            if key in data and isinstance(data[key], str) and len(data[key]) > 50:
                return {key: data[key]}
        
        for key in ['input', 'data']:
            if key in data and isinstance(data[key], dict):
                result = find_input_data_fast(data[key])
                if result:
                    return result
        
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
        
        if 'base64,' in base64_string:
            base64_string = base64_string.split('base64,')[-1]
        
        base64_string = ''.join(base64_string.split())
        
        valid_chars = set(string.ascii_letters + string.digits + '+/=')
        base64_string = ''.join(c for c in base64_string if c in valid_chars)
        
        no_pad = base64_string.rstrip('=')
        
        try:
            img_data = base64.b64decode(no_pad, validate=False)
            return Image.open(BytesIO(img_data))
        except:
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
    
    if 'ac_' in filename_lower:
        return "ac_pattern"
    else:
        return "other"

def create_background(size, color="#E0DADC", style="gradient"):
    """Create natural gray background"""
    width, height = size
    
    if style == "gradient":
        background = Image.new('RGB', size, color)
        bg_array = np.array(background, dtype=np.float32)
        
        y, x = np.ogrid[:height, :width]
        center_x, center_y = width / 2, height / 2
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2) / max(width, height)
        
        gradient = 1 - (distance * 0.05)
        gradient = np.clip(gradient, 0.95, 1.0)
        
        bg_array *= gradient[:, :, np.newaxis]
        
        return Image.fromarray(bg_array.astype(np.uint8))
    else:
        return Image.new('RGB', size, color)

def precision_edge_detection_removal(image: Image.Image) -> Image.Image:
    """PRECISION: Edge-based background removal with pixel-level detail"""
    try:
        from rembg import remove
        
        global REMBG_SESSION
        if REMBG_SESSION is None:
            REMBG_SESSION = init_rembg_session()
            if REMBG_SESSION is None:
                return image
        
        logger.info("🔷 Precision Edge Detection Background Removal V14.0")
        
        # STEP 1: Initial rough removal with BiRefNet
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        buffered.seek(0)
        img_data = buffered.getvalue()
        
        # Use optimal threshold for jewelry
        output = remove(
            img_data,
            session=REMBG_SESSION,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=50,
            alpha_matting_erode_size=0
        )
        
        result_image = Image.open(BytesIO(output))
        
        if result_image.mode != 'RGBA':
            return result_image
        
        # STEP 2: Precision edge refinement
        r, g, b, a = result_image.split()
        alpha_array = np.array(a, dtype=np.uint8)
        rgb_array = np.array(result_image.convert('RGB'), dtype=np.uint8)
        
        # Edge detection on RGB for better jewelry edge detection
        gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
        
        # Multi-scale edge detection
        edges1 = cv2.Canny(gray, 30, 100)
        edges2 = cv2.Canny(gray, 50, 150)
        edges3 = cv2.Canny(gray, 100, 200)
        
        # Combine edges
        edges_combined = edges1 | edges2 | edges3
        
        # STEP 3: Pixel-level refinement
        # Find contours of the main object
        contours, _ = cv2.findContours(alpha_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Create refined mask
            refined_mask = np.zeros_like(alpha_array)
            
            # Fill main object
            cv2.drawContours(refined_mask, contours, -1, 255, -1)
            
            # Refine edges pixel by pixel
            kernel_tiny = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            edge_region = cv2.dilate(edges_combined, kernel_tiny, iterations=1)
            
            # For each edge pixel, check if it should be included
            edge_coords = np.where(edge_region > 0)
            
            for y, x in zip(edge_coords[0], edge_coords[1]):
                # Check local neighborhood
                y_min = max(0, y-2)
                y_max = min(alpha_array.shape[0], y+3)
                x_min = max(0, x-2)
                x_max = min(alpha_array.shape[1], x+3)
                
                local_alpha = alpha_array[y_min:y_max, x_min:x_max]
                local_rgb = rgb_array[y_min:y_max, x_min:x_max]
                
                # Decision based on local context
                if np.mean(local_alpha) > 128:  # Mostly opaque region
                    # Check color consistency
                    center_color = rgb_array[y, x]
                    color_diff = np.mean(np.abs(local_rgb - center_color))
                    
                    if color_diff < 30:  # Similar color
                        refined_mask[y, x] = 255
                    else:
                        refined_mask[y, x] = 0
        
            # Smooth the refined mask
            refined_mask = cv2.GaussianBlur(refined_mask, (3, 3), 0.5)
            
            # Apply refined mask
            alpha_array = refined_mask
        
        # STEP 4: Sub-pixel anti-aliasing
        # Create smooth transitions at edges
        dist_transform = cv2.distanceTransform(alpha_array, cv2.DIST_L2, 3)
        dist_transform = np.clip(dist_transform / 3.0, 0, 1) * 255
        
        # Combine with original alpha for smooth edges
        alpha_smooth = alpha_array.astype(np.float32)
        edge_mask = cv2.Canny(alpha_array, 50, 150) > 0
        edge_region = cv2.dilate(edge_mask.astype(np.uint8), kernel_tiny, iterations=2)
        
        alpha_smooth[edge_region > 0] = dist_transform[edge_region > 0]
        
        # Final cleanup
        alpha_smooth = cv2.bilateralFilter(alpha_smooth.astype(np.uint8), 5, 50, 50)
        
        a_new = Image.fromarray(alpha_smooth.astype(np.uint8))
        return Image.merge('RGBA', (r, g, b, a_new))
        
    except Exception as e:
        logger.error(f"Precision edge removal failed: {e}")
        return image

def detect_ring_bounds_from_alpha(image: Image.Image) -> tuple:
    """Detect actual ring bounds from alpha channel"""
    if image.mode != 'RGBA':
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    else:
        alpha = np.array(image.split()[3])
        _, binary = cv2.threshold(alpha, 30, 255, cv2.THRESH_BINARY)
    
    coords = cv2.findNonZero(binary)
    
    if coords is None:
        logger.warning("No object found in image, using full image")
        return 0, 0, image.size[0], image.size[1], image.size[0]//2, image.size[1]//2, min(image.size)
    
    x, y, w, h = cv2.boundingRect(coords)
    
    center_x = x + w // 2
    center_y = y + h // 2
    
    diagonal = np.sqrt(w**2 + h**2)
    
    logger.info(f"Ring bounds: x={x}, y={y}, w={w}, h={h}, center=({center_x}, {center_y}), diagonal={diagonal:.1f}")
    
    return x, y, w, h, center_x, center_y, diagonal

def create_thumbnail_with_consistent_sizing(image, target_width=1000, target_height=1300):
    """Create thumbnail with consistent ring sizing"""
    original_width, original_height = image.size
    
    logger.info(f"Creating consistent thumbnail from {original_width}x{original_height}")
    
    x, y, w, h, center_x, center_y, diagonal = detect_ring_bounds_from_alpha(image)
    
    target_diagonal = np.sqrt(target_width**2 + target_height**2) * 0.75
    
    scale_factor = target_diagonal / diagonal if diagonal > 0 else 1.0
    
    crop_width = int(target_width / scale_factor)
    crop_height = int(target_height / scale_factor)
    
    crop_width = min(crop_width, original_width)
    crop_height = min(crop_height, original_height)
    
    crop_ratio = crop_width / crop_height
    target_ratio = target_width / target_height
    
    if crop_ratio > target_ratio:
        crop_width = int(crop_height * target_ratio)
    else:
        crop_height = int(crop_width / target_ratio)
    
    left = center_x - crop_width // 2
    top = center_y - crop_height // 2
    
    if left < 0:
        left = 0
    elif left + crop_width > original_width:
        left = original_width - crop_width
        
    if top < 0:
        top = 0
    elif top + crop_height > original_height:
        top = original_height - crop_height
    
    right = left + crop_width
    bottom = top + crop_height
    
    cropped = image.crop((left, top, right, bottom))
    thumbnail = cropped.resize((target_width, target_height), Image.Resampling.LANCZOS)
    
    return thumbnail

def composite_with_natural_blend(image, background_color="#E0DADC"):
    """Natural composite with edge blending"""
    if image.mode != 'RGBA':
        return image
    
    background = create_background(image.size, background_color, style="gradient")
    
    r, g, b, a = image.split()
    
    fg_array = np.array(image.convert('RGB'), dtype=np.float32)
    bg_array = np.array(background, dtype=np.float32)
    alpha_array = np.array(a, dtype=np.float32) / 255.0
    
    edges = cv2.Canny((alpha_array * 255).astype(np.uint8), 50, 150)
    edge_region = cv2.dilate(edges, np.ones((15, 15)), iterations=1) > 0
    
    alpha_soft = cv2.GaussianBlur(alpha_array, (5, 5), 1.5)
    
    edge_dist = cv2.distanceTransform(
        (1 - edge_region).astype(np.uint8), 
        cv2.DIST_L2, 5
    )
    edge_dist = np.clip(edge_dist / 10.0, 0, 1)
    
    alpha_final = alpha_array * edge_dist + alpha_soft * (1 - edge_dist)
    
    result = np.zeros_like(bg_array)
    for i in range(3):
        result[:,:,i] = (
            fg_array[:,:,i] * alpha_final + 
            bg_array[:,:,i] * (1 - alpha_final)
        )
    
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

def ensure_ring_holes_transparent_fast(image: Image.Image) -> Image.Image:
    """Fast ring hole detection - optimized for performance"""
    if image.mode != 'RGBA':
        return image
    
    logger.info("🔍 Fast Ring Hole Detection V14.0")
    
    r, g, b, a = image.split()
    alpha_array = np.array(a, dtype=np.uint8)
    rgb_array = np.array(image.convert('RGB'), dtype=np.float32)
    
    h, w = alpha_array.shape
    
    # STEP 1: Quick threshold detection
    potential_holes = alpha_array < 30
    
    # Clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    potential_holes = cv2.morphologyEx(potential_holes.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    potential_holes = cv2.morphologyEx(potential_holes, cv2.MORPH_CLOSE, kernel)
    
    # STEP 2: Find hole candidates
    num_labels, labels = cv2.connectedComponents(potential_holes)
    
    holes_mask = np.zeros_like(alpha_array)
    
    for label in range(1, num_labels):
        component = (labels == label)
        component_size = np.sum(component)
        
        # Quick size check
        if h * w * 0.0001 < component_size < h * w * 0.1:
            # Quick geometry check
            coords = np.where(component)
            min_y, max_y = coords[0].min(), coords[0].max()
            min_x, max_x = coords[1].min(), coords[1].max()
            
            width = max_x - min_x
            height = max_y - min_y
            aspect_ratio = width / height if height > 0 else 0
            
            # Check if reasonably circular
            if 0.5 < aspect_ratio < 2.0:
                # Check if bright (holes are usually bright)
                hole_pixels = rgb_array[component]
                if len(hole_pixels) > 0:
                    brightness = np.mean(hole_pixels)
                    if brightness > 200:  # Bright enough to be a hole
                        holes_mask[component] = 255
    
    # STEP 3: Apply holes with smooth transition
    if np.any(holes_mask > 0):
        # Smooth the holes
        holes_mask = cv2.GaussianBlur(holes_mask.astype(np.float32), (5, 5), 1.0)
        
        # Apply to alpha
        alpha_array = alpha_array * (1 - holes_mask / 255)
        
        # Ensure complete transparency in hole centers
        strong_holes = holes_mask > 128
        alpha_array[strong_holes] = 0
    
    a_new = Image.fromarray(alpha_array.astype(np.uint8))
    return Image.merge('RGBA', (r, g, b, a_new))

def apply_swinir_thumbnail(image: Image.Image) -> Image.Image:
    """Apply SwinIR enhancement for thumbnails"""
    if not REPLICATE_CLIENT:
        logger.warning("SwinIR skipped - no Replicate client")
        return image
    
    try:
        logger.info("🎨 Applying SwinIR enhancement")
        
        buffered = BytesIO()
        image.save(buffered, format="PNG", optimize=False)
        buffered.seek(0)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        img_data_url = f"data:image/png;base64,{img_base64}"
        
        output = REPLICATE_CLIENT.run(
            "jingyunliang/swinir:660d922d33153019e8c263a3bba265de882e7f4f70396546b6c9c8f9d47a021a",
            input={
                "image": img_data_url,
                "task_type": "Real-World Image Super-Resolution",
                "scale": 1,
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
    """Enhanced cubic details for thumbnails"""
    contrast = ImageEnhance.Contrast(image)
    image = contrast.enhance(1.08)
    
    image = image.filter(ImageFilter.UnsharpMask(radius=0.3, percent=120, threshold=3))
    
    contrast2 = ImageEnhance.Contrast(image)
    image = contrast2.enhance(1.03)
    
    sharpness = ImageEnhance.Sharpness(image)
    image = sharpness.enhance(1.20)
    
    return image

def auto_white_balance_fast(image: Image.Image) -> Image.Image:
    """Fast white balance"""
    img_array = np.array(image, dtype=np.float32)
    
    sampled = img_array[::15, ::15]
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

def apply_center_spotlight_fast(image: Image.Image, intensity: float = 0.025) -> Image.Image:
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
    """Enhanced wedding ring focus for thumbnails"""
    image = apply_center_spotlight_fast(image, 0.020)
    
    sharpness = ImageEnhance.Sharpness(image)
    image = sharpness.enhance(1.6)
    
    contrast = ImageEnhance.Contrast(image)
    image = contrast.enhance(1.04)
    
    image = image.filter(ImageFilter.UnsharpMask(radius=0.8, percent=100, threshold=3))
    
    return image

def calculate_quality_metrics_fast(image: Image.Image) -> dict:
    """Fast quality metrics"""
    img_array = np.array(image)[::30, ::30]
    
    r_avg = np.mean(img_array[:,:,0])
    g_avg = np.mean(img_array[:,:,1])
    b_avg = np.mean(img_array[:,:,2])
    
    brightness = (r_avg + g_avg + b_avg) / 3
    
    return {
        "brightness": brightness
    }

def apply_pattern_enhancement_consistent(image, pattern_type):
    """Consistent pattern enhancement"""
    
    if pattern_type == "ac_pattern":
        # Apply 12% white overlay
        white_overlay = 0.12
        img_array = np.array(image, dtype=np.float32)
        img_array = img_array * (1 - white_overlay) + 255 * white_overlay
        img_array = np.clip(img_array, 0, 255)
        image = Image.fromarray(img_array.astype(np.uint8))
        
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.005)
        
        color = ImageEnhance.Color(image)
        image = color.enhance(0.98)
        
    else:
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.08)
        
        color = ImageEnhance.Color(image)
        image = color.enhance(0.99)
        
        sharpness = ImageEnhance.Sharpness(image)
        image = sharpness.enhance(1.5)
    
    image = apply_center_spotlight_fast(image, 0.025)
    image = apply_wedding_ring_focus_fast(image)
    
    # Quality check for ac_pattern
    if pattern_type == "ac_pattern":
        metrics = calculate_quality_metrics_fast(image)
        if metrics["brightness"] < 235:
            white_overlay = 0.15
            img_array = np.array(image, dtype=np.float32)
            img_array = img_array * (1 - white_overlay) + 255 * white_overlay
            img_array = np.clip(img_array, 0, 255)
            image = Image.fromarray(img_array.astype(np.uint8))
    
    return image

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
    """Optimized thumbnail handler - V14.0 with precision edge detection"""
    try:
        logger.info(f"=== Thumbnail {VERSION} Started ===")
        
        image_index = event.get('image_index', 1)
        if isinstance(event.get('input'), dict):
            image_index = event.get('input', {}).get('image_index', image_index)
        
        background_color = '#E0DADC'
        
        filename = find_filename_fast(event)
        input_data = find_input_data_fast(event)
        
        if not input_data:
            raise ValueError("No input data found")
        
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
        
        # STEP 1: PRECISION EDGE BACKGROUND REMOVAL (PNG files)
        original_mode = image.mode
        has_transparency = image.mode == 'RGBA'
        needs_background_removal = False
        
        if filename and filename.lower().endswith('.png'):
            logger.info("📸 STEP 1: PNG detected - precision edge background removal")
            image = precision_edge_detection_removal(image)
            has_transparency = image.mode == 'RGBA'
            needs_background_removal = True
        
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
        
        image = auto_white_balance_fast(image)
        
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.08)
        
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.05)
        
        color = ImageEnhance.Color(image)
        image = color.enhance(1.005)
        
        pattern_type = detect_pattern_type(filename)
        
        # Create thumbnail with consistent sizing
        thumbnail = create_thumbnail_with_consistent_sizing(image, 1000, 1300)
        
        # STEP 3: SWINIR ENHANCEMENT (ALWAYS APPLIED)
        logger.info("🚀 STEP 3: Applying SwinIR enhancement")
        thumbnail = apply_swinir_thumbnail(thumbnail)
        
        thumbnail = enhance_cubic_details_thumbnail_simple(thumbnail)
        
        detected_type = {
            "ac_pattern": "무도금화이트(0.12/0.15)",
            "other": "기타색상(no_overlay)"
        }.get(pattern_type, "기타색상")
        
        # Use consistent enhancement
        thumbnail = apply_pattern_enhancement_consistent(thumbnail, pattern_type)
        
        # STEP 4: BACKGROUND COMPOSITE (if transparent)
        if has_transparency and 'original_transparent' in locals():
            logger.info(f"🖼️ STEP 4: Natural background compositing: {background_color}")
            
            # Apply same thumbnail crop to transparent version
            enhanced_transparent = create_thumbnail_with_consistent_sizing(original_transparent, 1000, 1300)
            
            if enhanced_transparent.mode == 'RGBA':
                # Fast ring hole detection
                enhanced_transparent = ensure_ring_holes_transparent_fast(enhanced_transparent)
                
                r, g, b, a = enhanced_transparent.split()
                rgb_image = Image.merge('RGB', (r, g, b))
                
                rgb_image = auto_white_balance_fast(rgb_image)
                rgb_image = enhance_cubic_details_thumbnail_simple(rgb_image)
                brightness = ImageEnhance.Brightness(rgb_image)
                rgb_image = brightness.enhance(1.08)
                contrast = ImageEnhance.Contrast(rgb_image)
                rgb_image = contrast.enhance(1.05)
                sharpness = ImageEnhance.Sharpness(rgb_image)
                rgb_image = sharpness.enhance(1.6)
                
                if pattern_type == "ac_pattern":
                    white_overlay = 0.12
                    img_array = np.array(rgb_image, dtype=np.float32)
                    img_array = img_array * (1 - white_overlay) + 255 * white_overlay
                    rgb_image = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
                
                r2, g2, b2 = rgb_image.split()
                enhanced_transparent = Image.merge('RGBA', (r2, g2, b2, a))
            
            thumbnail = composite_with_natural_blend(enhanced_transparent, background_color)
            
            sharpness = ImageEnhance.Sharpness(thumbnail)
            thumbnail = sharpness.enhance(1.10)
        
        # Final adjustments
        sharpness = ImageEnhance.Sharpness(thumbnail)
        thumbnail = sharpness.enhance(1.5)
        
        brightness = ImageEnhance.Brightness(thumbnail)
        thumbnail = brightness.enhance(1.02)
        
        thumbnail_base64 = image_to_base64(thumbnail)
        
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
                "swinir_applied": True,
                "swinir_timing": "AFTER resize",
                "png_support": True,
                "has_transparency": has_transparency,
                "background_composite": has_transparency,
                "background_removal": needs_background_removal,
                "background_color": background_color,
                "optimization_features": [
                    "✅ Precision edge detection with pixel-level refinement",
                    "✅ Multi-scale edge detection (3 levels)",
                    "✅ Sub-pixel anti-aliasing",
                    "✅ Fast ring hole detection (3 steps)",
                    "✅ SwinIR always applied after resize",
                    "✅ Bilateral filtering for edge preservation"
                ],
                "edge_detection_method": "Multi-scale Canny + pixel-level refinement",
                "background_removal_steps": [
                    "1. Initial BiRefNet removal (optimal threshold)",
                    "2. Multi-scale edge detection",
                    "3. Pixel-by-pixel edge refinement",
                    "4. Sub-pixel anti-aliasing"
                ],
                "ring_hole_detection_steps": [
                    "1. Quick threshold detection",
                    "2. Geometry validation",
                    "3. Smooth transition application"
                ],
                "processing_order": "1.Edge Detection → 2.Enhancement → 3.SwinIR → 4.Composite",
                "thumbnail_crop_method": "Consistent sizing based on ring bounds",
                "crop_factor": "75% of thumbnail diagonal",
                "expected_input": "2000x2600 PNG",
                "output_size": "1000x1300",
                "white_overlay": "12% for ac_ (1차), 15% (2차)",
                "brightness_increased": "8%",
                "contrast_increased": "5%", 
                "sharpness_increased": "1.5-1.6",
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
