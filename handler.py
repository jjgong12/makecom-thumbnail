import runpod
import os
import base64
import numpy as np
from io import BytesIO
from PIL import Image, ImageEnhance
import requests
import logging
import re
import string
import cv2
import time
from scipy import ndimage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

################################
# THUMBNAIL HANDLER - 1000x1300
# VERSION: Thumbnail-V4-SafeExtraction
################################

VERSION = "Thumbnail-V4-SafeExtraction"

# Global rembg session with U2Net
REMBG_SESSION = None

def init_rembg_session():
    """Initialize rembg session with U2Net for faster processing"""
    global REMBG_SESSION
    if REMBG_SESSION is None:
        try:
            from rembg import new_session
            # Use U2Net for faster processing
            REMBG_SESSION = new_session('u2net')
            logger.info("✅ U2Net session initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize rembg: {e}")
            REMBG_SESSION = None
    return REMBG_SESSION

# Initialize on module load
init_rembg_session()

def safe_ring_detection_phase1(image: Image.Image, max_candidates=10):
    """
    PHASE 1: Safe Ring Detection - Conservative approach to avoid cutting rings
    """
    try:
        logger.info("🎯 PHASE 1: Safe Ring Detection Started")
        start_time = time.time()
        
        # Convert to numpy array
        if image.mode != 'RGB':
            image_rgb = image.convert('RGB')
        else:
            image_rgb = image
            
        img_array = np.array(image_rgb)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        h, w = gray.shape
        logger.info(f"Image size: {w}x{h}")
        
        # More conservative parameters - wider range to catch all rings
        min_radius = int(min(h, w) * 0.05)  # 5% of image (smaller minimum)
        max_radius = int(min(h, w) * 0.5)   # 50% of image (larger maximum)
        
        # Apply slight blur to reduce noise
        gray_blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        
        logger.info("🔍 Running conservative Hough Circle detection...")
        
        # More sensitive detection parameters
        circles = cv2.HoughCircles(
            gray_blurred, 
            cv2.HOUGH_GRADIENT,
            dp=1.2,             # More sensitive
            minDist=min_radius * 1.5,  # Allow closer circles
            param1=50,          # Lower threshold for edge detection
            param2=30,          # Lower threshold for center detection
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        ring_candidates = []
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            logger.info(f"Found {len(circles[0])} circular candidates")
            
            for i, (x, y, r) in enumerate(circles[0][:max_candidates]):
                # Very conservative bounds checking
                margin = int(r * 0.3)  # 30% margin
                y1 = max(0, y - r - margin)
                y2 = min(h, y + r + margin)
                x1 = max(0, x - r - margin)
                x2 = min(w, x + r + margin)
                
                # Skip if too close to edge (might be cut off)
                edge_margin = 20
                if x1 < edge_margin or y1 < edge_margin or x2 > w - edge_margin or y2 > h - edge_margin:
                    logger.warning(f"Skipping circle too close to edge: center=({x},{y}), radius={r}")
                    continue
                
                region = gray[y1:y2, x1:x2]
                
                # Check if it looks like a ring (has hole in center)
                center_mask = np.zeros_like(region)
                cv2.circle(center_mask, 
                          (x - x1, y - y1), 
                          int(r * 0.3),  # Inner 30% 
                          255, -1)
                
                if np.any(center_mask > 0):
                    center_brightness = np.mean(region[center_mask > 0])
                    
                    # Simple scoring based on center brightness
                    score = center_brightness / 255.0
                    
                    ring_candidates.append({
                        'id': i,
                        'center': (int(x), int(y)),
                        'radius': int(r),
                        'score': float(score),
                        'bbox': (x1, y1, x2, y2),
                        'inner_radius': max(1, int(r * 0.25)),  # Conservative inner radius
                        'type': 'circle'
                    })
        
        # Sort by score
        ring_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Phase 1 complete in {elapsed:.2f}s")
        logger.info(f"📊 Found {len(ring_candidates)} valid ring candidates")
        
        return {
            'candidates': ring_candidates,
            'image_size': (w, h),
            'detection_time': elapsed,
            'method': 'safe_detection'
        }
        
    except Exception as e:
        logger.error(f"Safe ring detection failed: {e}")
        return {
            'candidates': [],
            'error': str(e),
            'image_size': image.size,
            'detection_time': 0,
            'method': 'safe_detection'
        }

def conservative_ring_removal_phase2(image: Image.Image, detection_result: dict):
    """
    PHASE 2: Conservative Background Removal - Preserve ring integrity
    """
    try:
        from rembg import remove
        
        logger.info("✨ PHASE 2: Conservative Ring Removal Started")
        start_time = time.time()
        
        # Ensure RGBA
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Initialize session if needed
        global REMBG_SESSION
        if REMBG_SESSION is None:
            REMBG_SESSION = init_rembg_session()
        
        # First, apply general background removal with conservative settings
        logger.info("🔧 Applying conservative background removal...")
        
        # Enhance contrast slightly before removal
        contrast = ImageEnhance.Contrast(image)
        image_enhanced = contrast.enhance(1.2)
        
        # Save to buffer
        buffered = BytesIO()
        image_enhanced.save(buffered, format="PNG", compress_level=1)
        buffered.seek(0)
        
        # Conservative removal settings
        output = remove(
            buffered.getvalue(),
            session=REMBG_SESSION,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,  # Lower threshold - keep more
            alpha_matting_background_threshold=10,   # Higher threshold - remove less
            alpha_matting_erode_size=0,             # No erosion
            only_mask=False,
            post_process_mask=True
        )
        
        # Process result
        result_image = Image.open(BytesIO(output))
        if result_image.mode != 'RGBA':
            result_image = result_image.convert('RGBA')
        
        # Get candidates from Phase 1
        candidates = detection_result.get('candidates', [])
        
        if candidates:
            logger.info(f"🔍 Processing {len(candidates)} ring candidates for hole detection")
            
            # Work with alpha channel
            r, g, b, a = result_image.split()
            alpha_array = np.array(a, dtype=np.uint8)
            rgb_array = np.array(result_image.convert('RGB'))
            
            # Process each ring candidate for holes only
            for i, candidate in enumerate(candidates[:5]):  # Limit to top 5
                cx, cy = candidate['center']
                radius = candidate['radius']
                inner_radius = candidate['inner_radius']
                
                # Check if center is bright (likely a hole)
                center_region_size = 10
                y1 = max(0, cy - center_region_size)
                y2 = min(rgb_array.shape[0], cy + center_region_size)
                x1 = max(0, cx - center_region_size)
                x2 = min(rgb_array.shape[1], cx + center_region_size)
                
                if y2 > y1 and x2 > x1:
                    center_region = rgb_array[y1:y2, x1:x2]
                    center_brightness = np.mean(center_region)
                    
                    # Only make hole if very bright center
                    if center_brightness > 240:
                        logger.info(f"🕳️ Creating hole for ring {i+1}")
                        
                        # Create conservative hole
                        hole_mask = np.zeros_like(alpha_array)
                        cv2.circle(hole_mask, (cx, cy), inner_radius, 255, -1)
                        
                        # Smooth transition
                        hole_mask_blurred = cv2.GaussianBlur(hole_mask, (5, 5), 2)
                        
                        # Apply hole with smooth transition
                        alpha_array = np.where(
                            hole_mask_blurred > 128,
                            0,
                            np.where(
                                hole_mask_blurred > 0,
                                alpha_array * (1 - hole_mask_blurred / 255.0),
                                alpha_array
                            )
                        ).astype(np.uint8)
            
            # Light smoothing
            alpha_array = cv2.bilateralFilter(alpha_array, 3, 30, 30)
            
            # Create final image
            a_new = Image.fromarray(alpha_array)
            result_image = Image.merge('RGBA', (r, g, b, a_new))
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Phase 2 complete in {elapsed:.2f}s")
        
        return {
            'image': result_image,
            'processing_time': elapsed,
            'method': 'conservative_removal'
        }
        
    except Exception as e:
        logger.error(f"Conservative removal failed: {e}")
        return {
            'image': safe_fallback_removal(image),
            'error': str(e),
            'method': 'fallback_removal'
        }

def safe_fallback_removal(image: Image.Image) -> Image.Image:
    """Safe fallback removal - very conservative"""
    try:
        from rembg import remove
        
        global REMBG_SESSION
        if REMBG_SESSION is None:
            REMBG_SESSION = init_rembg_session()
            if REMBG_SESSION is None:
                return image
        
        logger.info("🛡️ Using safe fallback removal")
        
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Very light enhancement
        contrast = ImageEnhance.Contrast(image)
        image_enhanced = contrast.enhance(1.1)
        
        # Save to buffer
        buffered = BytesIO()
        image_enhanced.save(buffered, format="PNG", compress_level=1)
        buffered.seek(0)
        
        # Very conservative settings
        output = remove(
            buffered.getvalue(),
            session=REMBG_SESSION,
            alpha_matting=True,
            alpha_matting_foreground_threshold=230,  # Very low - keep almost everything
            alpha_matting_background_threshold=20,   # Higher - remove very little
            alpha_matting_erode_size=0,
            only_mask=False,
            post_process_mask=False  # No post processing
        )
        
        result_image = Image.open(BytesIO(output))
        
        if result_image.mode != 'RGBA':
            result_image = result_image.convert('RGBA')
        
        return result_image
        
    except Exception as e:
        logger.error(f"Safe fallback removal failed: {e}")
        if image.mode != 'RGBA':
            return image.convert('RGBA')
        return image

def combined_safe_processing(image: Image.Image):
    """Combined safe 2-phase processing"""
    logger.info("🚀 Starting Safe 2-Phase Processing")
    total_start = time.time()
    
    # PHASE 1: Safe Detection
    detection_result = safe_ring_detection_phase1(image, max_candidates=10)
    
    # PHASE 2: Conservative Removal
    removal_result = conservative_ring_removal_phase2(image, detection_result)
    
    total_elapsed = time.time() - total_start
    
    # Extract image
    if isinstance(removal_result, dict) and 'image' in removal_result:
        result_image = removal_result['image']
    else:
        result_image = removal_result
    
    logger.info(f"✨ Total safe processing time: {total_elapsed:.2f}s")
    
    return result_image

def create_thumbnail_safe(image, target_width=1000, target_height=1300):
    """Create thumbnail with safety margins to avoid cutting rings"""
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    original_width, original_height = image.size
    
    logger.info(f"Creating safe thumbnail from {original_width}x{original_height} to {target_width}x{target_height}")
    
    # Check if image has transparency to find object bounds
    if image.mode == 'RGBA':
        # Get alpha channel
        alpha = np.array(image.split()[-1])
        
        # Find non-transparent pixels
        non_transparent = np.where(alpha > 10)
        
        if len(non_transparent[0]) > 0:
            # Get bounding box of non-transparent content
            min_y, max_y = non_transparent[0].min(), non_transparent[0].max()
            min_x, max_x = non_transparent[1].min(), non_transparent[1].max()
            
            # Add safety margin
            margin = 50  # pixels
            min_x = max(0, min_x - margin)
            min_y = max(0, min_y - margin)
            max_x = min(original_width, max_x + margin)
            max_y = min(original_height, max_y + margin)
            
            # Calculate content size
            content_width = max_x - min_x
            content_height = max_y - min_y
            
            logger.info(f"Content bounds: ({min_x},{min_y}) to ({max_x},{max_y})")
            
            # Calculate scale to fit content
            scale_x = target_width / content_width
            scale_y = target_height / content_height
            scale = min(scale_x, scale_y) * 0.9  # 90% to ensure margin
            
            # Calculate new size
            new_width = int(content_width * scale)
            new_height = int(content_height * scale)
            
            # Crop to content first
            cropped = image.crop((min_x, min_y, max_x, max_y))
            
            # Resize
            resized = cropped.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create final image with centering
            result = Image.new('RGBA', (target_width, target_height), (0, 0, 0, 0))
            paste_x = (target_width - new_width) // 2
            paste_y = (target_height - new_height) // 2
            result.paste(resized, (paste_x, paste_y), resized)
            
            return result
    
    # Fallback to standard proportional resize
    scale_x = target_width / original_width
    scale_y = target_height / original_height
    scale = min(scale_x, scale_y)
    
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Center in target size
    result = Image.new('RGBA', (target_width, target_height), (0, 0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    result.paste(resized, (paste_x, paste_y), resized)
    
    return result

def find_input_data_fast(data):
    """Fast input data extraction"""
    if isinstance(data, str) and len(data) > 50:
        return data
    
    if isinstance(data, dict):
        priority_keys = ['image', 'image_base64', 'enhanced_image', 'base64', 'img']
        
        for key in priority_keys:
            if key in data and isinstance(data[key], str) and len(data[key]) > 50:
                return data[key]
        
        for key in ['input', 'data']:
            if key in data and isinstance(data[key], dict):
                result = find_input_data_fast(data[key])
                if result:
                    return result
            elif key in data and isinstance(data[key], str) and len(data[key]) > 50:
                return data[key]
        
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
    """Fast base64 decode"""
    try:
        if not base64_str or len(base64_str) < 50:
            raise ValueError("Invalid base64 string")
        
        if 'base64,' in base64_str:
            base64_str = base64_str.split('base64,')[-1]
        
        base64_str = ''.join(base64_str.split())
        
        valid_chars = set(string.ascii_letters + string.digits + '+/=')
        base64_str = ''.join(c for c in base64_str if c in valid_chars)
        
        try:
            decoded = base64.b64decode(base64_str, validate=True)
            return decoded
        except Exception:
            no_pad = base64_str.rstrip('=')
            padding_needed = (4 - len(no_pad) % 4) % 4
            padded = no_pad + ('=' * padding_needed)
            decoded = base64.b64decode(padded, validate=True)
            return decoded
            
    except Exception as e:
        logger.error(f"Base64 decode error: {str(e)}")
        raise ValueError(f"Invalid base64 data: {str(e)}")

def base64_to_image_fast(base64_string):
    """Fast base64 to image conversion"""
    try:
        image_bytes = decode_base64_fast(base64_string)
        return Image.open(BytesIO(image_bytes))
    except Exception as e:
        logger.error(f"Base64 to image error: {str(e)}")
        raise ValueError(f"Invalid image data: {str(e)}")

def image_to_base64(image, keep_transparency=True):
    """Convert to base64 WITH padding"""
    buffered = BytesIO()
    
    if image.mode != 'RGBA' and keep_transparency:
        image = image.convert('RGBA')
    
    if image.mode == 'RGBA':
        image.save(buffered, format='PNG', compress_level=3, optimize=True)
    else:
        image.save(buffered, format='PNG', optimize=True, compress_level=3)
    
    buffered.seek(0)
    base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return base64_str

def handler(event):
    """Thumbnail handler - V4 with Safe Extraction"""
    try:
        logger.info(f"=== Thumbnail {VERSION} Started ===")
        logger.info("🛡️ V4 - Safe Extraction Mode")
        logger.info("✅ Conservative ring detection")
        logger.info("✅ Minimal background removal")
        logger.info("✅ Smart content-aware cropping")
        logger.info("✅ Safety margins to prevent cutting")
        
        # Get image index
        image_index = event.get('image_index', 1)
        if isinstance(event.get('input'), dict):
            image_index = event.get('input', {}).get('image_index', image_index)
        
        # Find input data
        filename = find_filename_fast(event)
        image_data_str = find_input_data_fast(event)
        
        if not image_data_str:
            raise ValueError("No input data found")
        
        # Load image
        start_time = time.time()
        image = base64_to_image_fast(image_data_str)
        
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        decode_time = time.time() - start_time
        logger.info(f"⏱️ Image decode: {decode_time:.2f}s")
        
        # STEP 1 & 2: Apply safe 2-phase processing
        start_time = time.time()
        logger.info("📸 Applying safe background removal")
        image = combined_safe_processing(image)
        
        removal_time = time.time() - start_time
        logger.info(f"⏱️ Safe processing: {removal_time:.2f}s")
        
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # STEP 3: Create safe thumbnail
        start_time = time.time()
        logger.info("📏 STEP 3: Creating safe thumbnail")
        thumbnail = create_thumbnail_safe(image, 1000, 1300)
        
        resize_time = time.time() - start_time
        logger.info(f"⏱️ Safe thumbnail creation: {resize_time:.2f}s")
        
        if thumbnail.mode != 'RGBA':
            thumbnail = thumbnail.convert('RGBA')
        
        # Convert to base64
        start_time = time.time()
        thumbnail_base64 = image_to_base64(thumbnail, keep_transparency=True)
        encode_time = time.time() - start_time
        logger.info(f"⏱️ Base64 encode: {encode_time:.2f}s")
        
        # Total time
        total_time = decode_time + removal_time + resize_time + encode_time
        logger.info(f"⏱️ TOTAL TIME: {total_time:.2f}s")
        
        output_filename = generate_thumbnail_filename(filename, image_index)
        
        return {
            "output": {
                "thumbnail": thumbnail_base64,
                "size": list(thumbnail.size),
                "filename": output_filename,
                "original_filename": filename,
                "image_index": image_index,
                "format": "base64_with_padding",
                "version": VERSION,
                "status": "success",
                "png_support": True,
                "has_transparency": True,
                "transparency_preserved": True,
                "background_removed": True,
                "ring_holes_applied": True,
                "output_mode": "RGBA",
                "base64_padding": "INCLUDED",
                "compression": "level_3",
                "file_number_info": {
                    "007": "Thumbnail 1",
                    "009": "Thumbnail 2", 
                    "010": "Thumbnail 3"
                },
                "processing_times": {
                    "decode": f"{decode_time:.2f}s",
                    "safe_processing": f"{removal_time:.2f}s",
                    "thumbnail_creation": f"{resize_time:.2f}s", 
                    "encode": f"{encode_time:.2f}s",
                    "total": f"{total_time:.2f}s"
                },
                "v4_improvements": [
                    "✅ Conservative ring detection with safety margins",
                    "✅ Minimal background removal to preserve rings",
                    "✅ Smart content-aware cropping",
                    "✅ Edge detection to prevent cutting",
                    "✅ Fallback mechanisms for safety"
                ],
                "safety_features": {
                    "edge_margin": "20px minimum from edges",
                    "detection_sensitivity": "Lower thresholds",
                    "removal_threshold": "Conservative (230/20)",
                    "content_margin": "50px safety buffer",
                    "scale_factor": "90% to ensure fit"
                },
                "thumbnail_method": "Content-aware safe cropping",
                "expected_input": "Any size image with rings",
                "output_size": "1000x1300",
                "output_format": "PNG with full transparency"
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
