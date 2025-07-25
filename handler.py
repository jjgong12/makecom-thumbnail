import runpod
import os
import base64
import numpy as np
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
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
# VERSION: New-Neo-V3-Shadow-Fix-Ultra-Enhanced-White5-Refined-Bright112
################################

VERSION = "New-Neo-V3-Shadow-Fix-Ultra-Enhanced-White5-Refined-Bright112"

# ===== GLOBAL INITIALIZATION =====
REPLICATE_API_TOKEN = os.environ.get('REPLICATE_API_TOKEN')
REPLICATE_CLIENT = None

if REPLICATE_API_TOKEN:
    try:
        REPLICATE_CLIENT = replicate.Client(api_token=REPLICATE_API_TOKEN)
        logger.info("✅ Replicate client initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Replicate: {e}")

# Global rembg session with U2Net
REMBG_SESSION = None

# Global font cache
KOREAN_FONT = None
FONT_VERIFIED = False

def init_rembg_session():
    """Initialize rembg session with U2Net for faster processing"""
    global REMBG_SESSION
    if REMBG_SESSION is None:
        try:
            from rembg import new_session
            # Use U2Net for faster processing
            REMBG_SESSION = new_session('u2net')
            logger.info("✅ U2Net session initialized for faster processing")
        except Exception as e:
            logger.error(f"❌ Failed to initialize rembg: {e}")
            REMBG_SESSION = None
    return REMBG_SESSION

# Initialize on module load
init_rembg_session()

def download_korean_font():
    """Download Korean font for text rendering - WITH CACHING"""
    global KOREAN_FONT, FONT_VERIFIED
    
    # Return cached font if already verified
    if KOREAN_FONT and FONT_VERIFIED:
        return KOREAN_FONT
    
    try:
        font_path = '/tmp/NanumGothic.ttf'
        
        # If font exists and not verified, verify it
        if os.path.exists(font_path) and not FONT_VERIFIED:
            try:
                # Test with actual Korean text
                test_font = ImageFont.truetype(font_path, 20, encoding='utf-8')
                img_test = Image.new('RGBA', (200, 100), (255, 255, 255, 0))
                draw_test = ImageDraw.Draw(img_test)
                # Test with various Korean characters
                test_text = "테스트 한글 폰트 확인"
                draw_test.text((10, 10), test_text, font=test_font, fill='black')
                logger.info("✅ Korean font verified and cached")
                KOREAN_FONT = font_path
                FONT_VERIFIED = True
                return font_path
            except Exception as e:
                logger.error(f"Font verification failed: {e}")
                os.remove(font_path)
                FONT_VERIFIED = False
        
        # Download if not exists or verification failed
        if not os.path.exists(font_path):
            font_urls = [
                'https://github.com/naver/nanumfont/raw/master/fonts/NanumFontSetup_TTF_GOTHIC/NanumGothic.ttf',
                'https://cdn.jsdelivr.net/gh/naver/nanumfont@master/fonts/NanumFontSetup_TTF_GOTHIC/NanumGothic.ttf',
                'https://github.com/naver/nanumfont/raw/master/fonts/NanumFontSetup_TTF_GOTHIC/NanumGothicBold.ttf'
            ]
            
            for url in font_urls:
                try:
                    logger.info(f"Downloading font from: {url}")
                    response = requests.get(url, timeout=30)
                    if response.status_code == 200 and len(response.content) > 100000:
                        with open(font_path, 'wb') as f:
                            f.write(response.content)
                        
                        # Verify the font works with Korean
                        test_font = ImageFont.truetype(font_path, 20, encoding='utf-8')
                        img_test = Image.new('RGBA', (200, 100), (255, 255, 255, 0))
                        draw_test = ImageDraw.Draw(img_test)
                        draw_test.text((10, 10), "한글 테스트", font=test_font, fill='black')
                        logger.info("✅ Korean font downloaded and verified successfully")
                        KOREAN_FONT = font_path
                        FONT_VERIFIED = True
                        return font_path
                except Exception as e:
                    logger.error(f"Failed to download from {url}: {e}")
                    continue
        
        logger.error("❌ Failed to download Korean font from all sources")
        return None
    except Exception as e:
        logger.error(f"Font download error: {e}")
        return None

def get_font(size, korean_font_path=None):
    """Get font with proper encoding - ENHANCED"""
    if korean_font_path and os.path.exists(korean_font_path):
        try:
            # Always use UTF-8 encoding for Korean fonts
            font = ImageFont.truetype(korean_font_path, size, encoding='utf-8')
            return font
        except Exception as e:
            logger.error(f"Font loading error: {e}")
    
    # Fallback to default
    try:
        logger.warning("Using default font as fallback")
        return ImageFont.load_default()
    except:
        return None

def safe_draw_text(draw, position, text, font, fill):
    """Safely draw text with proper encoding - ENHANCED"""
    try:
        if text and font:
            # Ensure text is properly encoded as UTF-8
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='replace')
            else:
                # Ensure it's a string and normalize
                text = str(text)
            
            # Draw the text
            draw.text(position, text, font=font, fill=fill)
    except Exception as e:
        logger.error(f"Text drawing error: {e}, text: {repr(text)}")
        # Fallback to simple text
        try:
            draw.text(position, "[Text Error]", font=font, fill=fill)
        except:
            pass

def get_text_size(draw, text, font):
    """Get text size compatible with different PIL versions"""
    try:
        # Ensure text is string
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='replace')
        else:
            text = str(text)
            
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        return draw.textsize(text, font=font)

def auto_crop_transparent(image):
    """Auto-crop transparent borders from image with padding"""
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    data = np.array(image)
    alpha = data[:,:,3]
    
    non_transparent = np.where(alpha > 10)
    
    if len(non_transparent[0]) == 0:
        return image
    
    min_y = non_transparent[0].min()
    max_y = non_transparent[0].max()
    min_x = non_transparent[1].min()
    max_x = non_transparent[1].max()
    
    padding = 10
    min_y = max(0, min_y - padding)
    max_y = min(data.shape[0] - 1, max_y + padding)
    min_x = max(0, min_x - padding)
    max_x = min(data.shape[1] - 1, max_x + padding)
    
    cropped = image.crop((min_x, min_y, max_x + 1, max_y + 1))
    
    # Ensure RGBA mode after crop
    if cropped.mode != 'RGBA':
        cropped = cropped.convert('RGBA')
    
    return cropped

def apply_enhanced_metal_color(image, metal_color, strength=0.3, color_id=""):
    """Apply enhanced metal color effect - Yellow/Rose/White/Antique Gold only"""
    # CRITICAL: Ensure RGBA mode
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    r, g, b, a = image.split()
    
    r_array = np.array(r, dtype=np.float32)
    g_array = np.array(g, dtype=np.float32)
    b_array = np.array(b, dtype=np.float32)
    a_array = np.array(a)
    
    mask = a_array > 0
    
    if mask.any():
        luminance = (0.299 * r_array + 0.587 * g_array + 0.114 * b_array) / 255.0
        
        metal_r, metal_g, metal_b = [c/255.0 for c in metal_color]
        
        if color_id == "white":
            brightness_boost = 1.05
            r_array[mask] = np.clip(r_array[mask] * brightness_boost, 0, 255)
            g_array[mask] = np.clip(g_array[mask] * brightness_boost, 0, 255)
            b_array[mask] = np.clip(b_array[mask] * brightness_boost, 0, 255)
        
        elif color_id == "rose":
            highlight_mask = luminance > 0.85
            shadow_mask = luminance < 0.15
            midtone_mask = ~highlight_mask & ~shadow_mask & mask
            
            if midtone_mask.any():
                blend_factor = 0.5
                r_array[midtone_mask] = r_array[midtone_mask] * (1 - blend_factor) + (255 * luminance[midtone_mask]) * blend_factor
                g_array[midtone_mask] = g_array[midtone_mask] * (1 - blend_factor) + (160 * luminance[midtone_mask]) * blend_factor
                b_array[midtone_mask] = b_array[midtone_mask] * (1 - blend_factor) + (120 * luminance[midtone_mask]) * blend_factor
            
            if highlight_mask.any():
                r_array[highlight_mask] = np.clip(r_array[highlight_mask] * 0.5 + 255 * 0.5, 0, 255)
                g_array[highlight_mask] = np.clip(g_array[highlight_mask] * 0.5 + 160 * 0.5, 0, 255)
                b_array[highlight_mask] = np.clip(b_array[highlight_mask] * 0.5 + 120 * 0.5, 0, 255)
            
            if shadow_mask.any():
                r_array[shadow_mask] = r_array[shadow_mask] * 0.8 + 50 * 0.2
                g_array[shadow_mask] = g_array[shadow_mask] * 0.8 + 30 * 0.2
                b_array[shadow_mask] = b_array[shadow_mask] * 0.8 + 20 * 0.2
        
        else:
            highlight_mask = luminance > 0.85
            shadow_mask = luminance < 0.15
            midtone_mask = ~highlight_mask & ~shadow_mask & mask
            
            if midtone_mask.any():
                blend_factor = strength * 2.0
                r_array[midtone_mask] = r_array[midtone_mask] * (1 - blend_factor) + (metal_r * 255 * luminance[midtone_mask]) * blend_factor
                g_array[midtone_mask] = g_array[midtone_mask] * (1 - blend_factor) + (metal_g * 255 * luminance[midtone_mask]) * blend_factor
                b_array[midtone_mask] = b_array[midtone_mask] * (1 - blend_factor) + (metal_b * 255 * luminance[midtone_mask]) * blend_factor
            
            if highlight_mask.any():
                tint_factor = strength * 0.5
                r_array[highlight_mask] = r_array[highlight_mask] * (1 - tint_factor) + (metal_r * 255) * tint_factor
                g_array[highlight_mask] = g_array[highlight_mask] * (1 - tint_factor) + (metal_g * 255) * tint_factor
                b_array[highlight_mask] = b_array[highlight_mask] * (1 - tint_factor) + (metal_b * 255) * tint_factor
            
            if shadow_mask.any():
                shadow_tint = strength * 0.2
                r_array[shadow_mask] = r_array[shadow_mask] * (1 - shadow_tint) + (metal_r * r_array[shadow_mask]) * shadow_tint
                g_array[shadow_mask] = g_array[shadow_mask] * (1 - shadow_tint) + (metal_g * g_array[shadow_mask]) * shadow_tint
                b_array[shadow_mask] = b_array[shadow_mask] * (1 - shadow_tint) + (metal_b * b_array[shadow_mask]) * shadow_tint
    
    r_array = np.clip(r_array, 0, 255)
    g_array = np.clip(g_array, 0, 255)
    b_array = np.clip(b_array, 0, 255)
    
    r_new = Image.fromarray(r_array.astype(np.uint8))
    g_new = Image.fromarray(g_array.astype(np.uint8))
    b_new = Image.fromarray(b_array.astype(np.uint8))
    
    # CRITICAL: Preserve alpha channel
    result = Image.merge('RGBA', (r_new, g_new, b_new, a))
    
    # Verify RGBA mode
    if result.mode != 'RGBA':
        logger.error("❌ WARNING: Metal color result is not RGBA!")
        result = result.convert('RGBA')
    
    return result

def create_color_section(ring_image, width=1200):
    """Create COLOR section with 4 metal variations - Yellow/Rose/White/Antique Gold only"""
    logger.info("Creating COLOR section with transparent PNGs")
    
    height = 850
    
    # Create section with WHITE background for display
    section_img = Image.new('RGB', (width, height), '#FFFFFF')
    draw = ImageDraw.Draw(section_img)
    
    korean_font_path = download_korean_font()
    title_font = get_font(56, korean_font_path)
    label_font = get_font(24, korean_font_path)
    
    # Title
    title = "COLOR"
    title_width, _ = get_text_size(draw, title, title_font)
    safe_draw_text(draw, (width//2 - title_width//2, 60), title, title_font, (40, 40, 40))
    
    # Remove background from ring image with ULTRA PRECISE V3 removal
    ring_no_bg = None
    if ring_image:
        try:
            logger.info("Removing background from ring image with ULTRA PRECISE V3 method")
            ring_no_bg = u2net_ultra_precise_removal_v3_shadow_fix_ultra_enhanced(ring_image)
            if ring_no_bg.mode != 'RGBA':
                ring_no_bg = ring_no_bg.convert('RGBA')
            ring_no_bg = auto_crop_transparent(ring_no_bg)
            logger.info("Background removed successfully with ultra precision V3")
        except Exception as e:
            logger.error(f"Failed to remove background: {e}")
            ring_no_bg = ring_image.convert('RGBA') if ring_image else None
    
    # Color definitions - ONLY 4 Gold types: Yellow/Rose/White/Antique
    colors = [
        ("yellow", "YELLOW GOLD", (255, 200, 50), 0.3),
        ("rose", "ROSE GOLD", (255, 160, 120), 0.35),
        ("white", "WHITE GOLD", (255, 255, 255), 0.0),
        ("antique", "ANTIQUE GOLD", (245, 235, 225), 0.1)
    ]
    
    # Grid layout
    grid_size = 260
    padding = 60
    start_x = (width - (grid_size * 2 + padding)) // 2
    start_y = 160
    
    for i, (color_id, label, color_rgb, strength) in enumerate(colors):
        row = i // 2
        col = i % 2
        
        x = start_x + col * (grid_size + padding)
        y = start_y + row * (grid_size + 100)
        
        # Create container with light background for visibility
        container = Image.new('RGBA', (grid_size, grid_size), (252, 252, 252, 255))
        container_draw = ImageDraw.Draw(container)
        
        # Border
        container_draw.rectangle([0, 0, grid_size-1, grid_size-1], 
                                fill=None, outline=(240, 240, 240), width=1)
        
        if ring_no_bg:
            try:
                # Copy ring and apply color
                ring_copy = ring_no_bg.copy()
                max_size = int(grid_size * 0.7)
                ring_copy.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
                # Apply color - PRESERVING TRANSPARENCY
                ring_tinted = apply_enhanced_metal_color(ring_copy, color_rgb, strength, color_id)
                
                # Center placement - preserving transparency
                paste_x = (grid_size - ring_tinted.width) // 2
                paste_y = (grid_size - ring_tinted.height) // 2
                
                # Paste with alpha channel
                container.paste(ring_tinted, (paste_x, paste_y), ring_tinted)
                
                logger.info(f"Applied {color_id} color with transparency preserved")
                
            except Exception as e:
                logger.error(f"Error applying color {color_id}: {e}")
        
        # Convert container to RGB for final section image
        container_rgb = Image.new('RGB', container.size, (252, 252, 252))
        container_rgb.paste(container, mask=container.split()[3] if container.mode == 'RGBA' else None)
        
        # Paste container to section image
        section_img.paste(container_rgb, (x, y))
        
        # Add label
        label_width, _ = get_text_size(draw, label, label_font)
        safe_draw_text(draw, (x + grid_size//2 - label_width//2, y + grid_size + 20), 
                     label, label_font, (80, 80, 80))
    
    logger.info(f"COLOR section created: {width}x{height}")
    return section_img

def u2net_ultra_precise_removal_v3_shadow_fix_ultra_enhanced(image: Image.Image) -> Image.Image:
    """ULTRA PRECISE V3 ENHANCED WITH REFINED EDGE PROCESSING"""
    try:
        from rembg import remove
        
        global REMBG_SESSION
        if REMBG_SESSION is None:
            REMBG_SESSION = init_rembg_session()
            if REMBG_SESSION is None:
                return image
        
        logger.info("🔷 U2Net ULTRA PRECISE V3 ENHANCED REFINED - Maximum Quality")
        
        # CRITICAL: Ensure RGBA mode before processing
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Enhanced pre-processing for jewelry with adaptive enhancement
        img_array = np.array(image, dtype=np.float32)
        
        # Adaptive contrast based on image statistics
        gray = cv2.cvtColor(img_array[:,:,:3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        
        # Dynamic contrast adjustment
        if std_val < 30:  # Low contrast image
            contrast_factor = 1.6
        elif std_val < 50:
            contrast_factor = 1.4
        else:
            contrast_factor = 1.2
        
        contrast = ImageEnhance.Contrast(image)
        image_enhanced = contrast.enhance(contrast_factor)
        
        # Apply slight sharpening for edge definition
        sharpness = ImageEnhance.Sharpness(image_enhanced)
        image_enhanced = sharpness.enhance(1.2)
        
        # Save image to buffer
        buffered = BytesIO()
        image_enhanced.save(buffered, format="PNG", compress_level=3, optimize=True)
        buffered.seek(0)
        img_data = buffered.getvalue()
        
        # Apply U2Net removal with REFINED settings for jewelry
        output = remove(
            img_data,
            session=REMBG_SESSION,
            alpha_matting=True,
            alpha_matting_foreground_threshold=350,  # Very high precision
            alpha_matting_background_threshold=0,
            alpha_matting_erode_size=0,  # No erosion
            only_mask=False,
            post_process_mask=True
        )
        
        result_image = Image.open(BytesIO(output))
        
        # CRITICAL: Ensure RGBA mode
        if result_image.mode != 'RGBA':
            result_image = result_image.convert('RGBA')
        
        # REFINED edge processing
        r, g, b, a = result_image.split()
        alpha_array = np.array(a, dtype=np.uint8)
        rgb_array = np.array(result_image.convert('RGB'), dtype=np.uint8)
        
        # Convert to float for processing
        alpha_float = alpha_array.astype(np.float32) / 255.0
        
        # STAGE 1: REFINED shadow detection with color spill analysis
        logger.info("🔍 REFINED shadow and color spill detection...")
        
        gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
        
        # Convert to multiple color spaces for comprehensive analysis
        hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
        h_channel, s_channel, v_channel = cv2.split(hsv)
        
        lab = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Multi-level shadow detection with refined thresholds
        # Level 1: Very faint shadows
        very_faint_shadows = (alpha_float > 0.01) & (alpha_float < 0.3)
        
        # Level 2: Low saturation gray areas with refined detection
        gray_shadows = (s_channel < 25) & (v_channel < 180) & (alpha_float < 0.6)
        
        # Level 3: Color spill detection
        # Detect green/blue screen spill
        green_spill = (h_channel > 35) & (h_channel < 85) & (s_channel > 30) & (alpha_float < 0.8)
        blue_spill = (h_channel > 85) & (h_channel < 135) & (s_channel > 30) & (alpha_float < 0.8)
        
        # Level 4: Edge-based shadow detection with multi-scale
        edges_fine = cv2.Canny(gray, 50, 150)
        edges_coarse = cv2.Canny(gray, 20, 80)
        edges_combined = edges_fine | edges_coarse
        
        # Dilate edges for shadow detection
        edge_dilated = cv2.dilate(edges_combined, np.ones((5,5)), iterations=1)
        edge_shadows = (alpha_float < 0.7) & (~edge_dilated.astype(bool))
        
        # Level 5: LAB-based shadow detection with tighter thresholds
        lab_shadows = (l_channel < 160) & (np.abs(a_channel - 128) < 15) & (np.abs(b_channel - 128) < 15)
        
        # Level 6: Gradient-based shadow detection
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        low_gradient = gradient_magnitude < np.percentile(gradient_magnitude, 10)
        gradient_shadows = low_gradient & (alpha_float < 0.5)
        
        # Combine all shadow detections
        all_shadows = (very_faint_shadows | gray_shadows | green_spill | blue_spill | 
                      edge_shadows | (lab_shadows & (alpha_float < 0.85)) | gradient_shadows)
        
        # REFINED shadow removal with feathering
        if np.any(all_shadows):
            logger.info("🔥 Removing shadows with refined feathering...")
            
            # Create distance map from main object
            main_object = (alpha_float > 0.8).astype(np.uint8)
            dist_from_object = cv2.distanceTransform(1 - main_object, cv2.DIST_L2, 5)
            
            # Feathered shadow removal based on distance
            shadow_removal_strength = np.clip(dist_from_object / 10, 0, 1)
            alpha_float[all_shadows] *= (1 - shadow_removal_strength[all_shadows])
        
        # STAGE 2: Ultra-precise multi-scale edge detection
        logger.info("🔍 Multi-scale edge detection with 8 methods...")
        
        # Method 1-3: Multi-scale Sobel
        sobel_scales = []
        for ksize in [3, 5, 7, 9]:
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
            sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
            sobel_scales.append(sobel_mag)
        
        sobel_combined = np.max(np.array(sobel_scales), axis=0)
        sobel_edges = (sobel_combined / sobel_combined.max() * 255).astype(np.uint8) > 20
        
        # Method 4: Scharr for fine details
        scharrx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        scharry = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
        scharr_magnitude = np.sqrt(scharrx**2 + scharry**2)
        scharr_edges = (scharr_magnitude / scharr_magnitude.max() * 255).astype(np.uint8) > 25
        
        # Method 5: Laplacian for jewelry details
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=5)
        laplacian_edges = np.abs(laplacian) > 20
        
        # Method 6-8: Multi-threshold Canny with non-maximum suppression
        canny_low = cv2.Canny(gray, 10, 40)
        canny_mid = cv2.Canny(gray, 30, 90)
        canny_high = cv2.Canny(gray, 50, 150)
        canny_ultra = cv2.Canny(gray, 80, 200)
        
        # Combine all edge detections
        all_edges = (sobel_edges | scharr_edges | laplacian_edges | 
                    (canny_low > 0) | (canny_mid > 0) | (canny_high > 0) | (canny_ultra > 0))
        
        # STAGE 3: Intelligent component analysis with shape metrics
        logger.info("🔍 Advanced component analysis with shape metrics...")
        
        # Binary mask for main object
        alpha_binary = (alpha_float > 0.5).astype(np.uint8)
        
        # Clean up with adaptive morphology
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        alpha_binary = cv2.morphologyEx(alpha_binary, cv2.MORPH_OPEN, kernel_open)
        alpha_binary = cv2.morphologyEx(alpha_binary, cv2.MORPH_CLOSE, kernel_close)
        
        num_labels, labels = cv2.connectedComponents(alpha_binary)
        
        if num_labels > 1:
            # Analyze all components with shape metrics
            component_stats = []
            
            for i in range(1, num_labels):
                component = (labels == i)
                size = np.sum(component)
                
                if size > 50:  # Minimum size threshold
                    # Calculate shape metrics
                    contours, _ = cv2.findContours(component.astype(np.uint8), 
                                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        contour = contours[0]
                        area = cv2.contourArea(contour)
                        perimeter = cv2.arcLength(contour, True)
                        
                        # Shape descriptors
                        if perimeter > 0:
                            circularity = (4 * np.pi * area) / (perimeter * perimeter)
                            hull = cv2.convexHull(contour)
                            hull_area = cv2.contourArea(hull)
                            solidity = area / hull_area if hull_area > 0 else 0
                            
                            # Eccentricity
                            if len(contour) >= 5:
                                ellipse = cv2.fitEllipse(contour)
                                (center, (width, height), angle) = ellipse
                                eccentricity = 0
                                if max(width, height) > 0:
                                    eccentricity = min(width, height) / max(width, height)
                            else:
                                eccentricity = 1
                            
                            # Distance from image center
                            component_center = np.mean(np.where(component), axis=1)
                            img_center = np.array([alpha_array.shape[0]/2, alpha_array.shape[1]/2])
                            dist_from_center = np.linalg.norm(component_center - img_center)
                            
                            component_stats.append({
                                'label': i,
                                'size': size,
                                'circularity': circularity,
                                'solidity': solidity,
                                'eccentricity': eccentricity,
                                'dist_from_center': dist_from_center,
                                'edge_ratio': np.sum(all_edges[component]) / size
                            })
            
            # Keep components based on comprehensive criteria
            if component_stats:
                # Sort by size
                component_stats.sort(key=lambda x: x['size'], reverse=True)
                
                main_size = component_stats[0]['size']
                min_component_size = max(100, main_size * 0.01)  # 1% of main object
                
                valid_components = []
                for stats in component_stats:
                    # Multi-criteria validation
                    size_valid = stats['size'] > min_component_size
                    shape_valid = (stats['solidity'] > 0.3 or stats['circularity'] > 0.2)
                    edge_valid = stats['edge_ratio'] > 0.1
                    
                    # Special case for very circular components (gems, holes)
                    is_circular = stats['circularity'] > 0.7
                    
                    if (size_valid and (shape_valid or edge_valid)) or is_circular:
                        valid_components.append(stats['label'])
                
                # Create final mask
                main_mask = np.zeros_like(alpha_binary, dtype=bool)
                for label_id in valid_components:
                    main_mask |= (labels == label_id)
                
                # Apply main mask with feathering
                alpha_float[~main_mask] = 0
        
        # STAGE 4: Refined artifact removal with texture analysis
        logger.info("🔍 Texture-based artifact removal...")
        
        # Calculate local texture metrics
        gray_float = gray.astype(np.float32)
        
        # Local standard deviation (texture measure)
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
        local_mean = cv2.filter2D(gray_float, -1, kernel)
        local_sq_mean = cv2.filter2D(gray_float**2, -1, kernel)
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0))
        
        # Low texture areas that might be artifacts
        low_texture = local_std < 5
        
        # Combined artifact detection
        artifacts = ((s_channel < 20) & (v_channel > 30) & (v_channel < 180) & 
                    (alpha_float > 0) & (alpha_float < 0.7) & low_texture)
        
        if np.any(artifacts):
            alpha_float[artifacts] = 0
        
        # STAGE 5: Advanced edge refinement with bilateral filtering
        logger.info("🔍 Advanced edge refinement...")
        
        # Apply guided filter for edge-aware smoothing
        alpha_uint8 = (alpha_float * 255).astype(np.uint8)
        
        # Bilateral filter to preserve edges while smoothing
        alpha_bilateral = cv2.bilateralFilter(alpha_uint8, 9, 50, 50)
        alpha_float = alpha_bilateral.astype(np.float32) / 255.0
        
        # Sharp edge enhancement with adaptive sigmoid
        edge_sharpness = np.zeros_like(alpha_float)
        
        # Calculate edge strength
        grad_alpha_x = cv2.Sobel(alpha_float, cv2.CV_64F, 1, 0, ksize=3)
        grad_alpha_y = cv2.Sobel(alpha_float, cv2.CV_64F, 0, 1, ksize=3)
        edge_strength = np.sqrt(grad_alpha_x**2 + grad_alpha_y**2)
        
        # Adaptive sigmoid based on edge strength
        high_edge_mask = edge_strength > 0.1
        low_edge_mask = ~high_edge_mask
        
        # Sharp sigmoid for strong edges
        k_sharp = 200
        threshold_sharp = 0.5
        edge_sharpness[high_edge_mask] = 1 / (1 + np.exp(-k_sharp * (alpha_float[high_edge_mask] - threshold_sharp)))
        
        # Softer sigmoid for weak edges
        k_soft = 50
        threshold_soft = 0.5
        edge_sharpness[low_edge_mask] = 1 / (1 + np.exp(-k_soft * (alpha_float[low_edge_mask] - threshold_soft)))
        
        alpha_float = edge_sharpness
        
        # STAGE 6: Final cleanup with morphological operations
        logger.info("🔍 Final morphological cleanup...")
        
        # Remove small holes
        alpha_binary_final = (alpha_float > 0.5).astype(np.uint8)
        
        # Fill small holes
        kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        alpha_filled = cv2.morphologyEx(alpha_binary_final, cv2.MORPH_CLOSE, kernel_fill)
        
        # Remove small components
        num_labels_final, labels_final = cv2.connectedComponents(alpha_filled)
        
        if num_labels_final > 2:
            sizes_final = [(i, np.sum(labels_final == i)) for i in range(1, num_labels_final)]
            if sizes_final:
                sizes_final.sort(key=lambda x: x[1], reverse=True)
                min_size = max(150, alpha_array.size * 0.0002)  # 0.02% of image
                
                valid_mask = np.zeros_like(alpha_filled, dtype=bool)
                for label_id, size in sizes_final:
                    if size > min_size:
                        valid_mask |= (labels_final == label_id)
                
                alpha_float[~valid_mask] = 0
        
        # Apply final smoothing
        alpha_final = cv2.GaussianBlur(alpha_float, (3, 3), 0.5)
        
        # Convert back to uint8
        alpha_array = np.clip(alpha_final * 255, 0, 255).astype(np.uint8)
        
        logger.info("✅ ULTRA PRECISE V3 ENHANCED REFINED complete")
        
        a_new = Image.fromarray(alpha_array)
        result = Image.merge('RGBA', (r, g, b, a_new))
        
        # Verify RGBA mode
        if result.mode != 'RGBA':
            logger.error("❌ WARNING: Result is not RGBA!")
            result = result.convert('RGBA')
        
        return result
        
    except Exception as e:
        logger.error(f"U2Net removal failed: {e}")
        # Ensure RGBA mode even on failure
        if image.mode != 'RGBA':
            return image.convert('RGBA')
        return image

def ensure_ring_holes_transparent_ultra_v3_enhanced(image: Image.Image) -> Image.Image:
    """ULTRA PRECISE V3 ENHANCED WITH REFINED HOLE DETECTION"""
    # CRITICAL: Preserve RGBA mode
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    logger.info("🔍 ULTRA PRECISE V3 ENHANCED REFINED Ring Hole Detection")
    
    r, g, b, a = image.split()
    alpha_array = np.array(a, dtype=np.uint8)
    rgb_array = np.array(image.convert('RGB'), dtype=np.uint8)
    
    h, w = alpha_array.shape
    
    # Convert to multiple color spaces
    hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv)
    
    lab = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    
    # STAGE 1: Comprehensive hole detection with refined thresholds
    # Multiple criteria for hole detection
    very_bright_v = v_channel > 248
    very_bright_l = l_channel > 243
    very_bright_gray = gray > 243
    
    # Very low saturation with adaptive threshold
    mean_saturation = np.mean(s_channel[alpha_array > 128])
    saturation_threshold = min(20, mean_saturation * 0.3)
    very_low_saturation = s_channel < saturation_threshold
    
    # Low color variance in LAB with adaptive threshold
    a_variance = np.std(a_channel[alpha_array > 128])
    b_variance = np.std(b_channel[alpha_array > 128])
    
    low_color_variance = ((np.abs(a_channel - 128) < min(20, a_variance)) & 
                         (np.abs(b_channel - 128) < min(20, b_variance)))
    
    # Alpha-based detection
    alpha_holes = alpha_array < 20
    
    # Combine all criteria
    potential_holes = ((very_bright_v | very_bright_l | very_bright_gray) & 
                      (very_low_saturation | low_color_variance)) | alpha_holes
    
    # STAGE 2: Advanced shape-based hole detection
    logger.info("🔍 Advanced shape and topology analysis...")
    
    if np.any(alpha_array > 128):
        # Distance transform from object
        object_mask = (alpha_array > 128).astype(np.uint8)
        dist_transform = cv2.distanceTransform(object_mask, cv2.DIST_L2, 5)
        
        # Multi-scale narrow region detection
        narrow_scales = []
        for scale in [15, 20, 25, 30]:
            narrow = (dist_transform > 1) & (dist_transform < scale)
            narrow_scales.append(narrow)
        
        narrow_regions = np.any(np.array(narrow_scales), axis=0)
        
        # Bright areas in narrow regions with gradient check
        gray_gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3,3)))
        narrow_bright = narrow_regions & ((gray > 235) | (v_channel > 240)) & (gray_gradient < 20)
        potential_holes |= narrow_bright
        
        # Topology-based hole detection
        # Find enclosed regions using flood fill
        inverted = cv2.bitwise_not(object_mask)
        num_inv_labels, inv_labels = cv2.connectedComponents(inverted)
        
        # Advanced enclosed region analysis
        for label in range(1, num_inv_labels):
            component = (inv_labels == label)
            if np.any(component):
                # Multi-criteria enclosure check
                component_uint8 = component.astype(np.uint8)
                
                # Method 1: Border touching
                dilated = cv2.dilate(component_uint8, np.ones((7,7)), iterations=1)
                touches_border = (np.any(dilated[0,:]) or np.any(dilated[-1,:]) or 
                                np.any(dilated[:,0]) or np.any(dilated[:,-1]))
                
                # Method 2: Convex hull analysis
                contours, _ = cv2.findContours(component_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    hull = cv2.convexHull(contours[0])
                    hull_mask = np.zeros_like(component_uint8)
                    cv2.fillPoly(hull_mask, [hull], 1)
                    
                    # Check if hull intersects with object
                    hull_intersects_object = np.any(hull_mask & object_mask)
                    
                    if not touches_border or hull_intersects_object:
                        # This is likely an enclosed hole
                        component_pixels = rgb_array[component]
                        if len(component_pixels) > 0:
                            # Multi-metric brightness analysis
                            brightness_rgb = np.mean(component_pixels)
                            brightness_v = np.mean(v_channel[component])
                            brightness_l = np.mean(l_channel[component])
                            brightness_percentile = np.percentile(gray[component], 90)
                            
                            if (brightness_rgb > 230 or brightness_v > 235 or 
                                brightness_l > 230 or brightness_percentile > 240):
                                potential_holes[component] = True
                                logger.info(f"Found enclosed bright region: RGB={brightness_rgb:.1f}, "
                                          f"V={brightness_v:.1f}, L={brightness_l:.1f}")
    
    # STAGE 3: Texture and pattern-based hole detection
    logger.info("🔍 Texture and pattern analysis...")
    
    # Local Binary Patterns for texture
    def compute_lbp(image, radius=1):
        rows, cols = image.shape
        lbp = np.zeros_like(image)
        
        for i in range(radius, rows - radius):
            for j in range(radius, cols - cols):
                center = image[i, j]
                binary_string = ''
                
                # 8 neighbors
                for angle in range(8):
                    x = int(i + radius * np.cos(2 * np.pi * angle / 8))
                    y = int(j + radius * np.sin(2 * np.pi * angle / 8))
                    
                    if 0 <= x < rows and 0 <= y < cols:
                        binary_string += '1' if image[x, y] >= center else '0'
                
                lbp[i, j] = int(binary_string, 2)
        
        return lbp
    
    # Compute LBP
    lbp = compute_lbp(gray)
    
    # Uniform texture areas (potential holes)
    lbp_variance = cv2.filter2D(lbp.astype(np.float32), -1, np.ones((5,5))/25)
    uniform_texture = lbp_variance < 10
    
    # Combine with brightness for hole detection
    texture_holes = uniform_texture & (gray > 220) & (alpha_array > 100)
    potential_holes |= texture_holes
    
    # Clean up noise with adaptive morphology
    kernel_size = max(3, min(7, int(np.sqrt(h * w) / 100)))
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    potential_holes = cv2.morphologyEx(potential_holes.astype(np.uint8), cv2.MORPH_OPEN, kernel_clean)
    potential_holes = cv2.morphologyEx(potential_holes, cv2.MORPH_CLOSE, kernel_clean)
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents(potential_holes)
    
    holes_mask = np.zeros_like(alpha_array, dtype=np.float32)
    
    # STAGE 4: Validate each hole candidate with enhanced criteria
    for label in range(1, num_labels):
        component = (labels == label)
        component_size = np.sum(component)
        
        # Adaptive size constraints based on image size
        min_size = max(15, h * w * 0.00001)  # 0.001%
        max_size = h * w * 0.3  # 30%
        
        if min_size < component_size < max_size:
            coords = np.where(component)
            if len(coords[0]) == 0:
                continue
            
            # Comprehensive component analysis
            component_pixels_rgb = rgb_array[component]
            component_alpha = alpha_array[component]
            
            if len(component_pixels_rgb) > 0:
                # Multi-space brightness analysis
                brightness_metrics = {
                    'rgb_mean': np.mean(component_pixels_rgb),
                    'rgb_max': np.max(component_pixels_rgb),
                    'v_mean': np.mean(v_channel[component]),
                    'v_percentile_90': np.percentile(v_channel[component], 90),
                    'l_mean': np.mean(l_channel[component]),
                    'gray_mean': np.mean(gray[component]),
                    'gray_median': np.median(gray[component])
                }
                
                # Saturation and color analysis
                saturation_metrics = {
                    'mean': np.mean(s_channel[component]),
                    'max': np.max(s_channel[component]),
                    'std': np.std(s_channel[component])
                }
                
                # Color uniformity in multiple spaces
                uniformity_metrics = {
                    'rgb_std': np.max(np.std(component_pixels_rgb, axis=0)),
                    'hsv_std': np.std(h_channel[component]),
                    'lab_a_std': np.std(a_channel[component]),
                    'lab_b_std': np.std(b_channel[component])
                }
                
                # Advanced shape analysis
                component_uint8 = component.astype(np.uint8) * 255
                contours, _ = cv2.findContours(component_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                shape_metrics = {}
                is_enclosed = False
                
                if contours:
                    contour = contours[0]
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)
                    
                    if perimeter > 0 and area > 0:
                        # Shape descriptors
                        shape_metrics['circularity'] = (4 * np.pi * area) / (perimeter * perimeter)
                        
                        hull = cv2.convexHull(contour)
                        hull_area = cv2.contourArea(hull)
                        shape_metrics['convexity'] = area / hull_area if hull_area > 0 else 0
                        shape_metrics['solidity'] = area / hull_area if hull_area > 0 else 0
                        
                        # Aspect ratio
                        if len(contour) >= 5:
                            ellipse = cv2.fitEllipse(contour)
                            (center, (width, height), angle) = ellipse
                            shape_metrics['aspect_ratio'] = min(width, height) / max(width, height) if max(width, height) > 0 else 1
                        else:
                            shape_metrics['aspect_ratio'] = 1
                        
                        # Enclosure check
                        x, y, w, h = cv2.boundingRect(contour)
                        if 'object_mask' in locals():
                            roi = object_mask[max(0,y-5):min(object_mask.shape[0],y+h+5), 
                                            max(0,x-5):min(object_mask.shape[1],x+w+5)]
                            if roi.shape[0] > 0 and roi.shape[1] > 0:
                                border_sum = (np.sum(roi[0,:]) + np.sum(roi[-1,:]) + 
                                            np.sum(roi[:,0]) + np.sum(roi[:,-1]))
                                expected_border = 2 * (roi.shape[0] + roi.shape[1]) - 4
                                if border_sum > expected_border * 0.7:
                                    is_enclosed = True
                
                # Multi-criteria confidence calculation
                confidence = 0.0
                
                # Brightness criteria (most important)
                brightness_score = 0.0
                if brightness_metrics['rgb_mean'] > 245 and brightness_metrics['v_mean'] > 248:
                    brightness_score = 0.5
                elif brightness_metrics['rgb_mean'] > 235 and brightness_metrics['v_percentile_90'] > 245:
                    brightness_score = 0.4
                elif brightness_metrics['gray_median'] > 240:
                    brightness_score = 0.3
                elif brightness_metrics['l_mean'] > 235:
                    brightness_score = 0.2
                
                confidence += brightness_score
                
                # Saturation criteria
                if saturation_metrics['mean'] < 8:
                    confidence += 0.3
                elif saturation_metrics['mean'] < 15 and saturation_metrics['std'] < 5:
                    confidence += 0.2
                elif saturation_metrics['max'] < 25:
                    confidence += 0.1
                
                # Color uniformity
                if uniformity_metrics['rgb_std'] < 8:
                    confidence += 0.2
                elif uniformity_metrics['rgb_std'] < 15:
                    confidence += 0.1
                
                # Shape criteria
                if shape_metrics:
                    shape_score = 0.0
                    if shape_metrics.get('circularity', 0) > 0.7:
                        shape_score += 0.1
                    if shape_metrics.get('convexity', 0) > 0.8:
                        shape_score += 0.05
                    if shape_metrics.get('aspect_ratio', 0) > 0.7:
                        shape_score += 0.05
                    
                    confidence += shape_score
                
                # Bonus for enclosed regions
                if is_enclosed:
                    confidence += 0.25
                
                # Alpha channel bonus
                if np.mean(component_alpha) < 200:
                    confidence += 0.1
                
                # Apply hole mask based on confidence
                if confidence > 0.45:  # Slightly lower threshold for better detection
                    holes_mask[component] = 255
                    logger.info(f"Hole detected: brightness={brightness_metrics['rgb_mean']:.1f}, "
                              f"saturation={saturation_metrics['mean']:.1f}, "
                              f"uniformity={uniformity_metrics['rgb_std']:.1f}, "
                              f"shape={shape_metrics.get('circularity', 0):.2f}, "
                              f"enclosed={is_enclosed}, confidence={confidence:.2f}")
    
    # STAGE 5: Apply holes with refined transitions
    if np.any(holes_mask > 0):
        # Create smooth transitions
        holes_mask_smooth = cv2.GaussianBlur(holes_mask, (7, 7), 1.5)
        
        # Create multiple transition zones
        kernel_sizes = [(3, 3), (5, 5), (7, 7)]
        transition_masks = []
        
        for ksize in kernel_sizes:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
            dilated = cv2.dilate(holes_mask, kernel, iterations=1)
            transition = (dilated > 0) & (holes_mask < 255)
            transition_masks.append(transition)
        
        alpha_float = alpha_array.astype(np.float32)
        
        # Apply holes with hard edge
        alpha_float[holes_mask_smooth > 240] = 0
        
        # Apply smooth transitions
        for i, transition in enumerate(transition_masks):
            if np.any(transition):
                # Distance-based transition with varying strength
                dist_from_hole = cv2.distanceTransform((holes_mask == 0).astype(np.uint8), cv2.DIST_L2, 3)
                transition_strength = 3 + i * 2  # Varying transition widths
                transition_alpha = np.clip(dist_from_hole / transition_strength, 0, 1)
                alpha_float[transition] *= transition_alpha[transition]
        
        # Final smoothing
        alpha_float = cv2.bilateralFilter(alpha_float.astype(np.uint8), 5, 50, 50).astype(np.float32)
        
        alpha_array = np.clip(alpha_float, 0, 255).astype(np.uint8)
        
        logger.info("✅ Ring holes applied with refined multi-scale transitions")
    
    a_new = Image.fromarray(alpha_array)
    result = Image.merge('RGBA', (r, g, b, a_new))
    
    # Verify RGBA mode
    if result.mode != 'RGBA':
        logger.error("❌ WARNING: Result is not RGBA!")
        result = result.convert('RGBA')
    
    return result

def process_color_section(job):
    """Process COLOR section special mode"""
    logger.info("Processing COLOR section special mode with ULTRA PRECISE V3 removal")
    
    try:
        # Find image data - FIXED
        image_data_str = find_input_data_fast(job)
        
        if not image_data_str:
            return {
                "output": {
                    "error": "No image data found for COLOR section",
                    "status": "error",
                    "version": VERSION
                }
            }
        
        # Decode and open image
        image_bytes = decode_base64_fast(image_data_str)
        ring_image = Image.open(BytesIO(image_bytes))
        
        # Create COLOR section
        color_section = create_color_section(ring_image, width=1200)
        
        # Convert to base64
        buffered = BytesIO()
        color_section.save(buffered, format="PNG", optimize=True, compress_level=3)
        buffered.seek(0)
        section_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        # FIXED: Keep padding for Google Script compatibility
        
        logger.info("COLOR section created successfully")
        
        return {
            "output": {
                "thumbnail": section_base64,
                "size": list(color_section.size),
                "section_type": "color",
                "special_mode": "color_section",
                "filename": "ac_wedding_011.png",
                "file_number": "011",
                "version": VERSION,
                "status": "success",
                "format": "base64_with_padding",
                "colors_generated": ["YELLOW GOLD", "ROSE GOLD", "WHITE GOLD", "ANTIQUE GOLD"],
                "background_removal": "ULTRA_PRECISE_V3_ENHANCED",
                "transparency_info": "Each ring variant has transparent background",
                "base64_padding": "INCLUDED",
                "compression": "level_3"
            }
        }
        
    except Exception as e:
        logger.error(f"Error creating COLOR section: {str(e)}")
        import traceback
        
        return {
            "output": {
                "error": str(e),
                "status": "error",
                "version": VERSION,
                "traceback": traceback.format_exc()
            }
        }

def find_input_data_fast(data):
    """FIXED: Fast input data extraction - consistent string return"""
    # Handle string input
    if isinstance(data, str) and len(data) > 50:
        return data
    
    # Handle dictionary input
    if isinstance(data, dict):
        priority_keys = ['image', 'image_base64', 'enhanced_image', 'base64', 'img']
        
        # Check priority keys first
        for key in priority_keys:
            if key in data and isinstance(data[key], str) and len(data[key]) > 50:
                return data[key]
        
        # Check nested structures
        for key in ['input', 'data']:
            if key in data and isinstance(data[key], dict):
                result = find_input_data_fast(data[key])
                if result:
                    return result
            elif key in data and isinstance(data[key], str) and len(data[key]) > 50:
                return data[key]
        
        # Check numbered keys as fallback
        for i in range(10):
            if str(i) in data and isinstance(data[str(i)], str) and len(data[str(i)]) > 50:
                return data[str(i)]
    
    return None

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
    """Generate thumbnail filename with fixed numbers"""
    if not original_filename:
        return f"thumbnail_{image_index:03d}.png"
    
    # Fixed thumbnail numbers: 007, 009, 010
    thumbnail_numbers = {1: "007", 2: "009", 3: "010"}
    
    new_filename = original_filename
    pattern = r'(_\d{3})'
    if re.search(pattern, new_filename):
        new_filename = re.sub(pattern, f'_{thumbnail_numbers.get(image_index, "007")}', new_filename)
    else:
        name_parts = new_filename.split('.')
        name_parts[0] += f'_{thumbnail_numbers.get(image_index, "007")}'
        new_filename = '.'.join(name_parts)
    
    return new_filename

def decode_base64_fast(base64_str: str) -> bytes:
    """ENHANCED: Fast base64 decode with consistent padding handling"""
    try:
        if not base64_str or len(base64_str) < 50:
            raise ValueError("Invalid base64 string")
        
        # Remove data URL prefix if present
        if 'base64,' in base64_str:
            base64_str = base64_str.split('base64,')[-1]
        
        # Clean whitespace
        base64_str = ''.join(base64_str.split())
        
        # Keep only valid base64 characters
        valid_chars = set(string.ascii_letters + string.digits + '+/=')
        base64_str = ''.join(c for c in base64_str if c in valid_chars)
        
        # FIXED: Try with padding first (normal base64)
        try:
            decoded = base64.b64decode(base64_str, validate=True)
            return decoded
        except Exception:
            # If fails, try to add proper padding
            no_pad = base64_str.rstrip('=')
            padding_needed = (4 - len(no_pad) % 4) % 4
            padded = no_pad + ('=' * padding_needed)
            decoded = base64.b64decode(padded, validate=True)
            return decoded
            
    except Exception as e:
        logger.error(f"Base64 decode error: {str(e)}")
        raise ValueError(f"Invalid base64 data: {str(e)}")

def base64_to_image_fast(base64_string):
    """ENHANCED: Fast base64 to image conversion with consistent handling"""
    try:
        image_bytes = decode_base64_fast(base64_string)
        return Image.open(BytesIO(image_bytes))
    except Exception as e:
        logger.error(f"Base64 to image error: {str(e)}")
        raise ValueError(f"Invalid image data: {str(e)}")

def detect_pattern_type(filename: str) -> str:
    """Detect pattern type - Updated with AB pattern"""
    if not filename:
        return "other"
    
    filename_lower = filename.lower()
    
    if 'ac_' in filename_lower:
        return "ac_pattern"
    elif 'ab_' in filename_lower:
        return "ab_pattern"
    else:
        return "other"

def create_thumbnail_proportional(image, target_width=1000, target_height=1300):
    """Create thumbnail with proper proportional sizing - preserving transparency"""
    # CRITICAL: Ensure RGBA mode
    if image.mode != 'RGBA':
        logger.warning(f"⚠️ Converting {image.mode} to RGBA in thumbnail creation")
        image = image.convert('RGBA')
    
    original_width, original_height = image.size
    
    logger.info(f"Creating proportional thumbnail from {original_width}x{original_height} to {target_width}x{target_height}")
    
    # For 2000x2600 -> 1000x1300, it's exactly 50% resize
    if original_width == 2000 and original_height == 2600:
        logger.info("Direct 50% resize for standard input size")
        result = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    else:
        # For other sizes, maintain aspect ratio
        scale_x = target_width / original_width
        scale_y = target_height / original_height
        scale = min(scale_x, scale_y)
        
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Resize first
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Center crop if needed - preserving transparency
        if new_width != target_width or new_height != target_height:
            # Create transparent background
            result = Image.new('RGBA', (target_width, target_height), (0, 0, 0, 0))
            
            # Center paste
            paste_x = (target_width - new_width) // 2
            paste_y = (target_height - new_height) // 2
            
            result.paste(resized, (paste_x, paste_y), resized)
        else:
            result = resized
    
    # Verify RGBA mode
    if result.mode != 'RGBA':
        logger.error("❌ WARNING: Thumbnail is not RGBA!")
        result = result.convert('RGBA')
    
    return result

def apply_swinir_thumbnail(image: Image.Image) -> Image.Image:
    """Apply SwinIR enhancement for thumbnails while preserving transparency"""
    if not REPLICATE_CLIENT:
        logger.warning("SwinIR skipped - no Replicate client")
        return image
    
    try:
        logger.info("🎨 Applying SwinIR enhancement with transparency support")
        
        # CRITICAL: Ensure RGBA mode
        if image.mode != 'RGBA':
            logger.warning(f"⚠️ Converting {image.mode} to RGBA for SwinIR")
            image = image.convert('RGBA')
        
        # Separate alpha channel
        r, g, b, a = image.split()
        rgb_image = Image.merge('RGB', (r, g, b))
        
        buffered = BytesIO()
        rgb_image.save(buffered, format="PNG", optimize=True, compress_level=3)
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
            
            # Recombine with alpha
            r2, g2, b2 = enhanced_image.split()
            result = Image.merge('RGBA', (r2, g2, b2, a))
            
            logger.info("✅ SwinIR enhancement successful with transparency")
            
            # Verify RGBA mode
            if result.mode != 'RGBA':
                logger.error("❌ WARNING: SwinIR result is not RGBA!")
                result = result.convert('RGBA')
            
            return result
            
    except Exception as e:
        logger.warning(f"SwinIR error: {str(e)}")
        
    return image

def auto_white_balance_fast(image: Image.Image) -> Image.Image:
    """Fast white balance - preserving transparency"""
    # CRITICAL: Ensure RGBA mode
    if image.mode != 'RGBA':
        logger.warning(f"⚠️ Converting {image.mode} to RGBA for white balance")
        image = image.convert('RGBA')
    
    r, g, b, a = image.split()
    rgb_img = Image.merge('RGB', (r, g, b))
    
    img_array = np.array(rgb_img, dtype=np.float32)
    
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
    
    rgb_balanced = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
    r2, g2, b2 = rgb_balanced.split()
    result = Image.merge('RGBA', (r2, g2, b2, a))
    
    # Verify RGBA mode
    if result.mode != 'RGBA':
        logger.error("❌ WARNING: White balance result is not RGBA!")
        result = result.convert('RGBA')
    
    return result

def apply_pattern_enhancement_transparent(image: Image.Image, pattern_type: str) -> Image.Image:
    """Apply pattern enhancement while TRULY preserving transparency - AC 20%, AB 16%, Other 5% - BRIGHTNESS 1.12"""
    # CRITICAL: Ensure RGBA mode
    if image.mode != 'RGBA':
        logger.warning(f"⚠️ Converting {image.mode} to RGBA in pattern enhancement")
        image = image.convert('RGBA')
    
    # CRITICAL: Process RGB channels separately to preserve alpha
    r, g, b, a = image.split()
    rgb_image = Image.merge('RGB', (r, g, b))
    
    # Convert to array for processing
    img_array = np.array(rgb_image, dtype=np.float32)
    
    # Apply enhancements based on pattern type - UPDATED WITH 5% FOR OTHER
    if pattern_type == "ac_pattern":
        logger.info("🔍 AC Pattern - Applying 20% white overlay with brightness 1.03")
        # Apply 20% white overlay
        white_overlay = 0.20
        img_array = img_array * (1 - white_overlay) + 255 * white_overlay
        img_array = np.clip(img_array, 0, 255)
        
        # Convert back to image
        rgb_image = Image.fromarray(img_array.astype(np.uint8))
        
        # UPDATED: Brightness increased by 0.01
        brightness = ImageEnhance.Brightness(rgb_image)
        rgb_image = brightness.enhance(1.03)  # Changed from 1.02
        
        color = ImageEnhance.Color(rgb_image)
        rgb_image = color.enhance(0.98)
        
        logger.info("✅ AC Pattern enhancement applied with 20% white overlay")
    
    elif pattern_type == "ab_pattern":
        logger.info("🔍 AB Pattern - Applying 16% white overlay and cool tone with brightness 1.03")
        # Apply 16% white overlay
        white_overlay = 0.16
        img_array = img_array * (1 - white_overlay) + 255 * white_overlay
        
        # Cool tone adjustment
        img_array[:,:,0] *= 0.96  # Reduce red
        img_array[:,:,1] *= 0.98  # Reduce green
        img_array[:,:,2] *= 1.02  # Increase blue
        
        # Cool color grading
        cool_overlay = np.array([240, 248, 255], dtype=np.float32)
        img_array = img_array * 0.95 + cool_overlay * 0.05
        
        img_array = np.clip(img_array, 0, 255)
        rgb_image = Image.fromarray(img_array.astype(np.uint8))
        
        color = ImageEnhance.Color(rgb_image)
        rgb_image = color.enhance(0.88)
        
        # UPDATED: Brightness increased by 0.01
        brightness = ImageEnhance.Brightness(rgb_image)
        rgb_image = brightness.enhance(1.03)  # Changed from 1.02
        
        logger.info("✅ AB Pattern enhancement applied with 16% white overlay")
        
    else:
        logger.info("🔍 Other Pattern - Applying 5% white overlay with brightness 1.12")
        # NEW: Apply 5% white overlay for other patterns
        white_overlay = 0.05
        img_array = img_array * (1 - white_overlay) + 255 * white_overlay
        img_array = np.clip(img_array, 0, 255)
        
        # Convert back to image
        rgb_image = Image.fromarray(img_array.astype(np.uint8))
        
        # BRIGHTNESS INCREASED TO 1.12
        brightness = ImageEnhance.Brightness(rgb_image)
        rgb_image = brightness.enhance(1.12)  # Increased from 1.09
        
        color = ImageEnhance.Color(rgb_image)
        rgb_image = color.enhance(0.99)
        
        # MATCHED WITH ENHANCEMENT: Use 1.5 for Other pattern
        sharpness = ImageEnhance.Sharpness(rgb_image)
        rgb_image = sharpness.enhance(1.5)
        
        logger.info("✅ Other Pattern enhancement applied with 5% white overlay and brightness 1.12")
    
    # UPDATED: Apply common enhancements with contrast 1.1
    contrast = ImageEnhance.Contrast(rgb_image)
    rgb_image = contrast.enhance(1.1)  # Changed from 1.06
    
    # Apply sharpening - EXACTLY SAME AS ENHANCEMENT HANDLER
    sharpness = ImageEnhance.Sharpness(rgb_image)
    rgb_image = sharpness.enhance(1.8)
    
    # CRITICAL: Recombine with ORIGINAL alpha channel
    r2, g2, b2 = rgb_image.split()
    enhanced_image = Image.merge('RGBA', (r2, g2, b2, a))
    
    logger.info(f"✅ Enhancement applied with contrast 1.1 and updated brightness. Mode: {enhanced_image.mode}")
    
    # Verify RGBA mode
    if enhanced_image.mode != 'RGBA':
        logger.error("❌ WARNING: Enhanced image is not RGBA!")
        enhanced_image = enhanced_image.convert('RGBA')
    
    return enhanced_image

def image_to_base64(image, keep_transparency=True):
    """Convert to base64 WITH padding - FIXED for Google Script compatibility"""
    buffered = BytesIO()
    
    # CRITICAL FIX: Force RGBA and save as PNG
    if image.mode != 'RGBA' and keep_transparency:
        logger.warning(f"⚠️ Converting {image.mode} to RGBA for transparency")
        image = image.convert('RGBA')
    
    if image.mode == 'RGBA':
        logger.info("💎 Saving RGBA image as PNG with compression level 3")
        image.save(buffered, format='PNG', compress_level=3, optimize=True)
    else:
        logger.info(f"Saving {image.mode} mode image as PNG")
        image.save(buffered, format='PNG', optimize=True, compress_level=3)
    
    buffered.seek(0)
    base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    # FIXED: Always return WITH padding for Google Script compatibility
    return base64_str

def handler(event):
    """Optimized thumbnail handler - New Neo V3 Shadow Fix Ultra Enhanced Refined - BRIGHTNESS 1.12"""
    try:
        logger.info(f"=== Thumbnail {VERSION} Started ===")
        logger.info("🎯 NEW NEO V3 REFINED: Enhanced Background Removal with Advanced Edge Processing")
        logger.info("💎 TRANSPARENT OUTPUT: Preserving alpha channel throughout")
        logger.info("🔥 ENHANCED FEATURES:")
        logger.info("  - Adaptive contrast based on image statistics")
        logger.info("  - Color spill detection (green/blue screen)")
        logger.info("  - Multi-scale edge detection (8 methods)")
        logger.info("  - Texture-based artifact removal with LBP")
        logger.info("  - Advanced shape metrics for component analysis")
        logger.info("  - Bilateral filtering for edge-aware smoothing")
        logger.info("  - Refined hole detection with multi-criteria scoring")
        logger.info("  - Feathered shadow removal based on distance")
        logger.info("🔧 AC PATTERN: 20% white overlay, brightness 1.03, contrast 1.1")
        logger.info("🔧 AB PATTERN: 16% white overlay, brightness 1.03, contrast 1.1")
        logger.info("✨ OTHER PATTERNS: 5% white overlay, brightness 1.12, contrast 1.1")
        logger.info("🎨 COLORS: Yellow/Rose/White/Antique Gold only")
        logger.info("🔄 PROCESSING ORDER: 1.Pattern Enhancement → 2.Resize → 3.SwinIR → 4.Ring Holes")
        logger.info("📌 BASE64 PADDING: ALWAYS INCLUDED for Google Script compatibility")
        logger.info("🗜️ COMPRESSION: Level 3 (balanced speed/size)")
        
        # Check for special mode first
        if event.get('special_mode') == 'color_section':
            return process_color_section(event)
        
        # Normal thumbnail processing with MATCHED ORDER
        image_index = event.get('image_index', 1)
        if isinstance(event.get('input'), dict):
            image_index = event.get('input', {}).get('image_index', image_index)
        
        filename = find_filename_fast(event)
        image_data_str = find_input_data_fast(event)
        
        if not image_data_str:
            raise ValueError("No input data found")
        
        # Load image using fixed function
        image = base64_to_image_fast(image_data_str)
        
        # CRITICAL: Convert to RGBA immediately
        if image.mode != 'RGBA':
            logger.info(f"Converting {image.mode} to RGBA immediately")
            image = image.convert('RGBA')
        
        # STEP 1: ALWAYS apply background removal with V3 ENHANCED REFINED
        logger.info("📸 STEP 1: ALWAYS applying ULTRA PRECISE V3 ENHANCED REFINED background removal")
        image = u2net_ultra_precise_removal_v3_shadow_fix_ultra_enhanced(image)
        
        # Verify RGBA after removal
        if image.mode != 'RGBA':
            logger.error("❌ Image lost RGBA after background removal!")
            image = image.convert('RGBA')
        
        # STEP 2: Apply white balance
        logger.info("⚖️ STEP 2: Applying white balance")
        image = auto_white_balance_fast(image)
        
        # STEP 3: PATTERN ENHANCEMENT FIRST (MATCHED ORDER)
        logger.info("🎨 STEP 3: Applying pattern enhancement FIRST (matched with Enhancement Handler)")
        pattern_type = detect_pattern_type(filename)
        
        detected_type = {
            "ac_pattern": "무도금화이트(0.20)",
            "ab_pattern": "무도금화이트-쿨톤(0.16)",
            "other": "기타색상(0.05)"
        }.get(pattern_type, "기타색상")
        
        # Apply pattern enhancement with UPDATED settings
        image = apply_pattern_enhancement_transparent(image, pattern_type)
        
        # STEP 4: RESIZE (MATCHED ORDER)
        logger.info("📏 STEP 4: Creating proportional thumbnail")
        thumbnail = create_thumbnail_proportional(image, 1000, 1300)
        
        # STEP 5: SWINIR ENHANCEMENT (MATCHED ORDER)
        logger.info("🚀 STEP 5: Applying SwinIR enhancement")
        thumbnail = apply_swinir_thumbnail(thumbnail)
        
        # STEP 6: Ultra precise V3 ENHANCED ring hole detection (MATCHED ORDER)
        logger.info("🔍 STEP 6: Applying ULTRA PRECISE V3 ENHANCED REFINED ring hole detection")
        thumbnail = ensure_ring_holes_transparent_ultra_v3_enhanced(thumbnail)
        
        # Final verification
        if thumbnail.mode != 'RGBA':
            logger.error("❌ CRITICAL: Final thumbnail is not RGBA! Converting...")
            thumbnail = thumbnail.convert('RGBA')
        
        # CRITICAL: NO BACKGROUND COMPOSITE - Keep transparency
        logger.info("💎 NO background composite - keeping pure transparency")
        
        logger.info(f"✅ Final thumbnail mode: {thumbnail.mode}")
        logger.info(f"✅ Final thumbnail size: {thumbnail.size}")
        
        # Convert to base64 - WITH padding for Google Script
        thumbnail_base64 = image_to_base64(thumbnail, keep_transparency=True)
        
        # Verify transparency is preserved
        logger.info("✅ Transparency preserved in final output")
        
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
                "format": "base64_with_padding",
                "version": VERSION,
                "status": "success",
                "swinir_applied": True,
                "swinir_timing": "AFTER pattern enhancement and resize",
                "png_support": True,
                "has_transparency": True,
                "transparency_preserved": True,
                "background_removed": True,
                "background_applied": False,
                "output_mode": "RGBA",
                "base64_padding": "INCLUDED",
                "compression": "level_3",
                "special_modes_available": ["color_section"],
                "file_number_info": {
                    "007": "Thumbnail 1",
                    "009": "Thumbnail 2", 
                    "010": "Thumbnail 3",
                    "011": "COLOR section"
                },
                "contrast_brightness_update": {
                    "contrast": "1.1",
                    "brightness_ac_ab": "1.03",
                    "brightness_other": "1.12 (increased from 1.09)",
                    "white_overlay_ac": "20%",
                    "white_overlay_ab": "16%",
                    "white_overlay_other": "5%",
                    "reason": "Brightness increased to 1.12 for better visibility"
                },
                "refined_background_removal_features": [
                    "✅ ADAPTIVE CONTRAST: Dynamic adjustment based on image statistics",
                    "✅ COLOR SPILL DETECTION: Green/blue screen removal",
                    "✅ MULTI-SCALE EDGE DETECTION: 8 methods including 4 Sobel scales",
                    "✅ TEXTURE ANALYSIS: Local Binary Patterns for artifact detection",
                    "✅ ADVANCED SHAPE METRICS: Circularity, solidity, eccentricity, aspect ratio",
                    "✅ FEATHERED SHADOW REMOVAL: Distance-based strength adjustment",
                    "✅ BILATERAL FILTERING: Edge-aware smoothing",
                    "✅ ADAPTIVE SIGMOID: Different sharpness for strong/weak edges",
                    "✅ MULTI-SCALE TRANSITIONS: 3 levels of hole edge smoothing",
                    "✅ CONFIDENCE SCORING: Multi-criteria validation (0.45 threshold)",
                    "✅ TOPOLOGY ANALYSIS: Convex hull for enclosed region detection",
                    "✅ GRADIENT-BASED SHADOWS: Low gradient area detection",
                    "✅ MORPHOLOGICAL CLEANUP: Adaptive kernel sizes",
                    "✅ COMPONENT VALIDATION: Edge ratio and shape descriptors",
                    "✅ TEXTURE UNIFORMITY: LBP variance for hole detection"
                ],
                "thumbnail_method": "Proportional resize (no aggressive cropping)",
                "processing_order": "1.U2Net-Ultra-V3-Enhanced-Refined → 2.White Balance → 3.Pattern Enhancement → 4.Resize → 5.SwinIR → 6.Ring Holes",
                "edge_detection": "ULTRA PRECISE V3 ENHANCED REFINED (8-method combination)",
                "korean_support": "ENHANCED with font caching",
                "expected_input": "2000x2600 (any format)",
                "output_size": "1000x1300",
                "output_format": "PNG with full transparency",
                "transparency_info": "Full RGBA transparency preserved - NO background or shadows",
                "white_overlay": "AC: 20% | AB: 16% | Other: 5%",
                "brightness_adjustments": "AC/AB: 1.03 | Other: 1.12",
                "contrast_final": "1.1",
                "sharpness_final": "Other: 1.5 → Final: 1.8",
                "quality": "95",
                "google_script_compatibility": "Base64 WITH padding - FIXED",
                "metal_colors": "Yellow Gold, Rose Gold, White Gold, Antique Gold",
                "enhancement_matching": "FULLY MATCHED with Enhancement Handler V3 Enhanced",
                "shadow_elimination": "REFINED with feathering and multi-level detection",
                "brightness_update": "Other pattern brightness increased to 1.12 for better visibility"
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
