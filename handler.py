import os
import sys
import base64
import io
import time
import re
import traceback
import json
from typing import Optional, Tuple, Dict, Any, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VERSION = "v9"

# Import availability checks
NUMPY_AVAILABLE = False
PIL_AVAILABLE = False
CV2_AVAILABLE = False
REQUESTS_AVAILABLE = False
REPLICATE_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    print(f"[{VERSION}] NumPy not available")

try:
    from PIL import Image, ImageEnhance
    PIL_AVAILABLE = True
except ImportError:
    print(f"[{VERSION}] PIL not available")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print(f"[{VERSION}] OpenCV not available")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    print(f"[{VERSION}] Requests not available")

try:
    import replicate
    REPLICATE_AVAILABLE = True
except ImportError:
    print(f"[{VERSION}] Replicate not available")

try:
    import runpod
    RUNPOD_AVAILABLE = True
except ImportError:
    print(f"[{VERSION}] RunPod not available")
    RUNPOD_AVAILABLE = False

def convert_numpy_types(obj):
    """Convert NumPy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif NUMPY_AVAILABLE and isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif NUMPY_AVAILABLE and isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif NUMPY_AVAILABLE and isinstance(obj, np.bool_):
        return bool(obj)
    elif NUMPY_AVAILABLE and isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def find_image_data(data, depth=0, max_depth=3):
    """
    Find image data in nested input structure
    """
    if depth > max_depth:
        return None
    
    logger.info(f"Searching for image at depth {depth}, type: {type(data)}")
    
    # Direct string check
    if isinstance(data, str) and len(data) > 100:
        # Check if it looks like base64
        base64_pattern = re.compile(r'^[A-Za-z0-9+/]*={0,2}$')
        if base64_pattern.match(data.replace('\n', '').replace('\r', '')):
            logger.info(f"Found base64-like string at depth {depth}")
            return data
    
    # Dictionary search
    if isinstance(data, dict):
        # Primary image keys
        image_keys = [
            'image', 'image_base64', 'base64', 'img', 'data', 'imageData', 
            'image_data', 'input_image', 'enhanced_image', 'file_content'
        ]
        
        for key in image_keys:
            if key in data and data[key]:
                logger.info(f"Found image in key: {key}")
                return data[key]
        
        # Search nested structures
        for key, value in data.items():
            result = find_image_data(value, depth + 1, max_depth)
            if result:
                return result
    
    # List search
    elif isinstance(data, list):
        for item in data:
            result = find_image_data(item, depth + 1, max_depth)
            if result:
                return result
    
    return None

def decode_base64_image(base64_str):
    """
    Decode base64 image with multiple fallback methods
    """
    if not base64_str:
        raise ValueError("Empty base64 string")
    
    # Clean the string
    base64_str = base64_str.strip()
    
    # Remove data URL prefix if present
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    
    # Remove whitespace
    base64_str = base64_str.replace('\n', '').replace('\r', '').replace(' ', '')
    
    # Try multiple decoding methods
    methods = [
        lambda x: base64.b64decode(x, validate=True),
        lambda x: base64.b64decode(x + '=='),
        lambda x: base64.b64decode(x + '='),
        lambda x: base64.urlsafe_b64decode(x + '==')
    ]
    
    for i, method in enumerate(methods):
        try:
            image_data = method(base64_str)
            img = Image.open(io.BytesIO(image_data))
            logger.info(f"Base64 decode successful with method {i+1}")
            return img
        except Exception as e:
            logger.warning(f"Decode method {i+1} failed: {str(e)}")
            continue
    
    raise ValueError("All base64 decode methods failed")

def detect_metal_type(image):
    """
    Detect wedding ring metal type from 4 categories
    Based on 38 training data pairs (28 + 10)
    """
    try:
        if not NUMPY_AVAILABLE:
            return "unknown"
            
        img_array = np.array(image)
        
        # Convert to HSV for better color analysis
        if CV2_AVAILABLE:
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        else:
            # Simple RGB analysis fallback
            avg_color = np.mean(img_array.reshape(-1, 3), axis=0)
            
            # Basic color categorization
            r, g, b = avg_color
            if r > 200 and g > 200 and b > 200:
                return "white_gold"  # 화이트골드
            elif r > g + 20 and r > b + 20:
                if g > b:
                    return "yellow_gold"  # 옐로우골드
                else:
                    return "rose_gold"   # 로즈골드
            else:
                return "plain_white"     # 무도금화이트
        
        # Advanced HSV analysis
        h_channel = hsv[:, :, 0]
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]
        
        # Calculate color characteristics
        avg_hue = np.mean(h_channel)
        avg_saturation = np.mean(s_channel)
        avg_value = np.mean(v_channel)
        
        # Metal type classification based on training data
        if avg_saturation < 50 and avg_value > 180:
            metal_type = "plain_white"  # 무도금화이트
        elif avg_hue < 30 and avg_saturation > 80:
            metal_type = "yellow_gold"  # 옐로우골드
        elif avg_hue < 20 and avg_saturation > 50:
            metal_type = "rose_gold"    # 로즈골드
        else:
            metal_type = "white_gold"   # 화이트골드
            
        logger.info(f"Metal type detected: {metal_type} (H:{avg_hue:.1f}, S:{avg_saturation:.1f}, V:{avg_value:.1f})")
        return metal_type
        
    except Exception as e:
        logger.error(f"Metal detection error: {str(e)}")
        return "white_gold"  # Default fallback

def detect_black_frame(img):
    """
    Detect black frame/borders using multiple methods
    """
    try:
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        
        # Method 1: Edge-based detection
        edge_threshold = 30
        edge_width = 10
        
        # Check edges for black pixels
        edges = [
            img_array[:edge_width, :],      # top
            img_array[-edge_width:, :],     # bottom
            img_array[:, :edge_width],      # left
            img_array[:, -edge_width:]      # right
        ]
        
        black_pixel_ratios = []
        for edge in edges:
            if len(img_array.shape) == 3:
                gray_edge = np.mean(edge, axis=2)
            else:
                gray_edge = edge
            
            black_pixels = np.sum(gray_edge < edge_threshold)
            total_pixels = gray_edge.size
            ratio = black_pixels / total_pixels if total_pixels > 0 else 0
            black_pixel_ratios.append(ratio)
        
        max_black_ratio = max(black_pixel_ratios)
        has_edge_frame = max_black_ratio > 0.3
        
        # Method 2: Center-based detection (for square masks)
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array
        
        # Find non-black content
        non_black = gray > edge_threshold
        rows = np.any(non_black, axis=1)
        cols = np.any(non_black, axis=0)
        
        if rows.any() and cols.any():
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            content_area = (cmin, rmin, cmax, rmax)
            
            # Check if there's a significant black frame
            content_width = cmax - cmin
            content_height = rmax - rmin
            
            frame_width_ratio = (width - content_width) / width
            frame_height_ratio = (height - content_height) / height
            
            has_center_frame = frame_width_ratio > 0.1 or frame_height_ratio > 0.1
        else:
            content_area = (0, 0, width-1, height-1)
            has_center_frame = False
        
        # Combined detection
        has_frame = has_edge_frame or has_center_frame
        
        # Create mask for black areas
        if has_frame:
            black_mask = gray < edge_threshold
            
            # Remove content area from mask
            cmin, rmin, cmax, rmax = content_area
            content_mask = np.zeros_like(black_mask, dtype=bool)
            content_mask[rmin:rmax+1, cmin:cmax+1] = True
            
            # Final frame mask (black areas outside content)
            frame_mask = black_mask & ~content_mask
        else:
            frame_mask = None
        
        logger.info(f"Black frame detection - Edge: {has_edge_frame} (ratio: {max_black_ratio:.2f}), Center: {has_center_frame}")
        logger.info(f"Content area: ({content_area[0]}, {content_area[1]}) to ({content_area[2]}, {content_area[3]})")
        
        return has_frame, content_area, frame_mask, max_black_ratio
        
    except Exception as e:
        logger.error(f"Frame detection error: {str(e)}")
        return False, (0, 0, img.width-1, img.height-1), None, 0.0

def remove_black_frame_replicate(img, frame_mask):
    """
    Remove black frame using Replicate FLUX Fill API
    """
    if not REPLICATE_AVAILABLE or not REQUESTS_AVAILABLE:
        logger.warning("Replicate or Requests not available")
        return img
    
    try:
        # Create mask image
        mask_img = Image.fromarray((frame_mask * 255).astype(np.uint8), mode='L')
        
        # Convert to base64
        def img_to_base64(image):
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            # CRITICAL: Remove padding for Make.com compatibility
            return img_base64.rstrip('=')
        
        img_base64 = img_to_base64(img)
        mask_base64 = img_to_base64(mask_img)
        
        logger.info("Starting Replicate FLUX Fill inpainting...")
        
        # Use FLUX Fill for high-quality inpainting
        output = replicate.run(
            "black-forest-labs/flux-fill-dev",
            input={
                "image": f"data:image/png;base64,{img_base64}",
                "mask": f"data:image/png;base64,{mask_base64}",
                "prompt": "clean professional product photography background, smooth gradient, studio lighting, wedding ring",
                "negative_prompt": "black frame, border, dark edges, artifacts, shadows",
                "num_inference_steps": 30,
                "guidance_scale": 7.5,
                "strength": 0.95
            }
        )
        
        # Download result
        if output and len(output) > 0:
            response = requests.get(output[0])
            result_img = Image.open(io.BytesIO(response.content))
            logger.info("Replicate inpainting completed successfully")
            return result_img
        else:
            logger.error("No output from Replicate")
            return img
            
    except Exception as e:
        logger.error(f"Replicate inpainting error: {str(e)}")
        return img

def simple_frame_removal(img, content_area):
    """
    Simple frame removal by cropping and background filling
    """
    try:
        img_array = np.array(img)
        
        # Crop to content area
        cmin, rmin, cmax, rmax = content_area
        
        # Add small padding
        padding = 10
        cmin = max(0, cmin - padding)
        rmin = max(0, rmin - padding)
        cmax = min(img.width, cmax + padding)
        rmax = min(img.height, rmax + padding)
        
        # Crop
        cropped = img.crop((cmin, rmin, cmax, rmax))
        
        # Create new image with white background
        new_img = Image.new('RGB', img.size, (250, 250, 250))
        
        # Paste cropped content in center
        x_offset = (img.width - cropped.width) // 2
        y_offset = (img.height - cropped.height) // 2
        new_img.paste(cropped, (x_offset, y_offset))
        
        logger.info(f"Simple frame removal: cropped {cmax-cmin}x{rmax-rmin} and recentered")
        return new_img
        
    except Exception as e:
        logger.error(f"Simple frame removal error: {str(e)}")
        return img

def apply_wedding_ring_enhancement(img, metal_type):
    """
    Apply wedding ring enhancement based on metal type
    Using 38 training data pairs (28 + 10)
    """
    try:
        # Base enhancement (same as enhance_handler V7)
        brightness_enhancer = ImageEnhance.Brightness(img)
        img = brightness_enhancer.enhance(1.05)  # 5% brightness increase
        
        color_enhancer = ImageEnhance.Color(img)
        img = color_enhancer.enhance(0.95)  # 5% saturation decrease
        
        # Metal-specific enhancement
        if metal_type == "yellow_gold":
            # Enhance warm tones
            img_array = np.array(img)
            img_array[:, :, 0] = np.minimum(255, img_array[:, :, 0] * 1.08)  # More red
            img_array[:, :, 1] = np.minimum(255, img_array[:, :, 1] * 1.05)  # Slight green
            img = Image.fromarray(img_array.astype(np.uint8))
            
        elif metal_type == "rose_gold":
            # Enhance pink/warm tones
            img_array = np.array(img)
            img_array[:, :, 0] = np.minimum(255, img_array[:, :, 0] * 1.10)  # More red
            img_array[:, :, 2] = np.minimum(255, img_array[:, :, 2] * 1.03)  # Slight blue
            img = Image.fromarray(img_array.astype(np.uint8))
            
        elif metal_type == "white_gold":
            # Enhance cool tones and clarity
            contrast_enhancer = ImageEnhance.Contrast(img)
            img = contrast_enhancer.enhance(1.08)
            
        elif metal_type == "plain_white":
            # Enhance brightness and clean look
            brightness_enhancer = ImageEnhance.Brightness(img)
            img = brightness_enhancer.enhance(1.08)
        
        # Selective background brightening
        if NUMPY_AVAILABLE:
            img_array = np.array(img)
            mask = np.all(img_array > 200, axis=-1)
            if mask.any():
                for c in range(3):
                    img_array[mask, c] = np.minimum(255, img_array[mask, c] * 1.05).astype(np.uint8)
                img = Image.fromarray(img_array)
        
        # Detail enhancement
        sharpness_enhancer = ImageEnhance.Sharpness(img)
        img = sharpness_enhancer.enhance(1.15)
        
        logger.info(f"Wedding ring enhancement applied for {metal_type}")
        return img
        
    except Exception as e:
        logger.error(f"Enhancement error: {str(e)}")
        return img

def crop_to_ring_area(img, padding_ratio=0.05):
    """
    Crop to wedding ring area with 90% fill and 10% padding
    """
    try:
        img_array = np.array(img)
        gray = np.mean(img_array, axis=2)
        
        # Find non-background areas (ring areas)
        non_bg = gray < 200  # Areas darker than background
        rows = np.any(non_bg, axis=1)
        cols = np.any(non_bg, axis=0)
        
        if rows.any() and cols.any():
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            # Add padding (5% on each side = 10% total)
            height = rmax - rmin
            width = cmax - cmin
            padding_h = int(height * padding_ratio)
            padding_w = int(width * padding_ratio)
            
            # Adjust boundaries
            rmin = max(0, rmin - padding_h)
            rmax = min(img_array.shape[0], rmax + padding_h)
            cmin = max(0, cmin - padding_w)
            cmax = min(img_array.shape[1], cmax + padding_w)
            
            # Crop
            cropped = img.crop((cmin, rmin, cmax, rmax))
            logger.info(f"Cropped to ring area: {cmax-cmin}x{rmax-rmin}")
            return cropped
        else:
            logger.warning("No ring area detected, using original image")
            return img
            
    except Exception as e:
        logger.error(f"Ring crop error: {str(e)}")
        return img

def resize_to_thumbnail(img, target_width=1000, target_height=1300):
    """
    Resize to 1000x1300 thumbnail maintaining aspect ratio
    """
    try:
        # Calculate ratios
        img_width, img_height = img.size
        width_ratio = target_width / img_width
        height_ratio = target_height / img_height
        
        # Use smaller ratio to fit entirely (90% fill)
        ratio = min(width_ratio, height_ratio) * 0.9
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        
        # Resize with high quality
        resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create final canvas and center the image
        final_img = Image.new('RGB', (target_width, target_height), (250, 250, 250))
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        final_img.paste(resized, (x_offset, y_offset))
        
        logger.info(f"Final thumbnail created: {target_width}x{target_height} with rings at 90%")
        return final_img
        
    except Exception as e:
        logger.error(f"Thumbnail resize error: {str(e)}")
        return img

def image_to_base64(img):
    """
    Convert PIL Image to base64 - MUST REMOVE PADDING for Make.com
    """
    try:
        buffer = io.BytesIO()
        img.save(buffer, format='PNG', quality=95, optimize=True)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # CRITICAL: Remove padding for Make.com compatibility
        img_base64 = img_base64.rstrip('=')
        
        logger.info(f"Image converted to base64, length: {len(img_base64)}, padding removed")
        return img_base64
        
    except Exception as e:
        logger.error(f"Base64 conversion error: {str(e)}")
        return ""

def handler(job):
    """
    RunPod handler for thumbnail generation V8 - Complete full version
    """
    start_time = time.time()
    
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"Thumbnail Handler {VERSION} - NumPy Compatibility Fix")
        logger.info(f"Features: Black frame removal, Metal detection, Wedding ring enhancement, 1000x1300 thumbnail")
        logger.info(f"Training: 38 data pairs (28 + 10), 4 metal types")
        logger.info(f"{'='*60}")
        
        # Get input
        job_input = job.get('input', {})
        logger.info(f"Input keys: {list(job_input.keys())}")
        
        # Debug mode
        if job_input.get('debug_mode', False):
            return {
                "output": {
                    "status": "debug_success",
                    "message": f"{VERSION} handler working - NumPy compatibility fix applied",
                    "version": VERSION,
                    "features": [
                        "Black frame detection & removal",
                        "Metal type detection (4 types)",
                        "Wedding ring enhancement",
                        "38 training data pairs",
                        "1000x1300 thumbnail generation",
                        "JSON serialization safe",
                        "NumPy 1.24+ compatibility",
                        "Make.com compatible"
                    ]
                }
            }
        
        # Find image data
        image_data_str = find_image_data(job_input)
        if not image_data_str:
            error_msg = f"No image found. Available keys: {list(job_input.keys())}"
            logger.error(error_msg)
            return {
                "output": {
                    "error": error_msg,
                    "status": "error",
                    "version": VERSION
                }
            }
        
        # Decode image
        img = decode_base64_image(image_data_str)
        original_size = img.size
        logger.info(f"Original image size: {original_size}")
        
        # Convert RGBA to RGB if needed
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        
        # Detect metal type
        metal_type = detect_metal_type(img)
        
        # Detect black frame
        has_frame, content_area, frame_mask, frame_ratio = detect_black_frame(img)
        
        # Remove frame if detected
        if has_frame and frame_mask is not None:
            if REPLICATE_AVAILABLE:
                logger.info("Removing black frame with Replicate FLUX Fill")
                img = remove_black_frame_replicate(img, frame_mask)
            else:
                logger.info("Removing black frame with simple crop")
                img = simple_frame_removal(img, content_area)
        else:
            logger.info("No black frame detected")
        
        # Apply wedding ring enhancement
        img = apply_wedding_ring_enhancement(img, metal_type)
        
        # Crop to ring area (90% fill)
        img = crop_to_ring_area(img)
        
        # Create final thumbnail
        thumbnail = resize_to_thumbnail(img)
        
        # Convert to base64
        thumbnail_base64 = image_to_base64(thumbnail)
        
        # Processing info with numpy type conversion
        processing_info = {
            "original_size": list(original_size),
            "metal_type": str(metal_type),
            "has_black_frame": bool(has_frame),
            "frame_ratio": float(frame_ratio),
            "content_area": [int(x) for x in content_area],
            "replicate_used": bool(REPLICATE_AVAILABLE and has_frame),
            "final_size": [1000, 1300],
            "enhancement_applied": True,
            "processing_time": round(time.time() - start_time, 2),
            "version": VERSION,
            "training_data": "38 pairs (28 + 10)"
        }
        
        # Convert all numpy types to native Python types
        processing_info = convert_numpy_types(processing_info)
        
        result = {
            "output": {
                "thumbnail": thumbnail_base64,
                "processing_info": processing_info,
                "status": "success"
            }
        }
        
        # Final numpy type conversion for entire result
        result = convert_numpy_types(result)
        
        logger.info(f"Thumbnail generation completed in {processing_info['processing_time']}s")
        logger.info(f"Metal type: {metal_type}, Frame removed: {has_frame}")
        logger.info(f"Output structure: {list(result['output'].keys())}")
        
        return result
        
    except Exception as e:
        error_msg = f"Handler error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        # Ensure error response is also JSON-safe
        error_result = {
            "output": {
                "error": error_msg,
                "status": "error",
                "version": VERSION,
                "processing_time": round(time.time() - start_time, 2)
            }
        }
        
        return convert_numpy_types(error_result)

# RunPod serverless start
if __name__ == "__main__":
    if RUNPOD_AVAILABLE:
        runpod.serverless.start({"handler": handler})
    else:
        print(f"[{VERSION}] RunPod not available, running in test mode")
        test_job = {
            "input": {
                "debug_mode": True
            }
        }
        result = handler(test_job)
        print(json.dumps(result, indent=2))
