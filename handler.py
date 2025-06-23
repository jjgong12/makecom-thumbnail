import os
import sys
import base64
import io
import time
import re
import traceback
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VERSION = "v10"

# Safe imports
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    print(f"[{VERSION}] NumPy not available")
    NUMPY_AVAILABLE = False

try:
    from PIL import Image, ImageEnhance
    PIL_AVAILABLE = True
except ImportError:
    print(f"[{VERSION}] PIL not available")
    PIL_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print(f"[{VERSION}] OpenCV not available")
    CV2_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    print(f"[{VERSION}] Requests not available")
    REQUESTS_AVAILABLE = False

try:
    import replicate
    REPLICATE_AVAILABLE = True
except ImportError:
    print(f"[{VERSION}] Replicate not available")
    REPLICATE_AVAILABLE = False

try:
    import runpod
    RUNPOD_AVAILABLE = True
except ImportError:
    print(f"[{VERSION}] RunPod not available")
    RUNPOD_AVAILABLE = False

def safe_json_convert(obj):
    """
    Safely convert objects to JSON-serializable types
    COMPLETELY AVOIDS np.bool references
    """
    if obj is None:
        return None
    elif isinstance(obj, bool):
        return bool(obj)  # Native Python bool
    elif isinstance(obj, int):
        return int(obj)
    elif isinstance(obj, float):
        return float(obj)
    elif isinstance(obj, str):
        return str(obj)
    elif isinstance(obj, list):
        return [safe_json_convert(item) for item in obj]
    elif isinstance(obj, tuple):
        return [safe_json_convert(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(key): safe_json_convert(value) for key, value in obj.items()}
    elif NUMPY_AVAILABLE and hasattr(np, 'ndarray') and isinstance(obj, np.ndarray):
        return obj.tolist()
    elif NUMPY_AVAILABLE and hasattr(np, 'integer') and isinstance(obj, np.integer):
        return int(obj)
    elif NUMPY_AVAILABLE and hasattr(np, 'floating') and isinstance(obj, np.floating):
        return float(obj)
    elif NUMPY_AVAILABLE and str(type(obj)).startswith("<class 'numpy.bool"):
        return bool(obj)  # Handle any numpy bool type safely
    else:
        # Fallback: convert to string
        return str(obj)

def find_image_data(data, depth=0, max_depth=3):
    """Find image data in nested input structure"""
    if depth > max_depth:
        return None
    
    logger.info(f"Searching for image at depth {depth}, type: {type(data)}")
    
    if isinstance(data, str) and len(data) > 100:
        base64_pattern = re.compile(r'^[A-Za-z0-9+/]*={0,2}$')
        if base64_pattern.match(data.replace('\n', '').replace('\r', '')):
            logger.info(f"Found base64-like string at depth {depth}")
            return data
    
    if isinstance(data, dict):
        image_keys = [
            'image', 'image_base64', 'base64', 'img', 'data', 'imageData', 
            'image_data', 'input_image', 'enhanced_image', 'file_content'
        ]
        
        for key in image_keys:
            if key in data and data[key]:
                logger.info(f"Found image in key: {key}")
                return data[key]
        
        for key, value in data.items():
            result = find_image_data(value, depth + 1, max_depth)
            if result:
                return result
    
    elif isinstance(data, list):
        for item in data:
            result = find_image_data(item, depth + 1, max_depth)
            if result:
                return result
    
    return None

def decode_base64_image(base64_str):
    """Decode base64 image with multiple fallback methods"""
    if not base64_str:
        raise ValueError("Empty base64 string")
    
    base64_str = base64_str.strip()
    
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    
    base64_str = base64_str.replace('\n', '').replace('\r', '').replace(' ', '')
    
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
    """Detect wedding ring metal type"""
    try:
        if not NUMPY_AVAILABLE:
            return "white_gold"
            
        img_array = np.array(image)
        avg_color = np.mean(img_array.reshape(-1, 3), axis=0)
        r, g, b = avg_color
        
        if r > 200 and g > 200 and b > 200:
            return "white_gold"
        elif r > g + 20 and r > b + 20:
            if g > b:
                return "yellow_gold"
            else:
                return "rose_gold"
        else:
            return "plain_white"
        
    except Exception as e:
        logger.error(f"Metal detection error: {str(e)}")
        return "white_gold"

def detect_black_frame(img):
    """Detect black frame/borders"""
    try:
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        
        edge_threshold = 30
        edge_width = 10
        
        edges = [
            img_array[:edge_width, :],
            img_array[-edge_width:, :], 
            img_array[:, :edge_width],
            img_array[:, -edge_width:]
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
        has_frame = max_black_ratio > 0.3
        
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array
        
        non_black = gray > edge_threshold
        rows = np.any(non_black, axis=1)
        cols = np.any(non_black, axis=0)
        
        if rows.any() and cols.any():
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            content_area = (cmin, rmin, cmax, rmax)
        else:
            content_area = (0, 0, width-1, height-1)
        
        frame_mask = None
        if has_frame:
            black_mask = gray < edge_threshold
            cmin, rmin, cmax, rmax = content_area
            content_mask = np.zeros_like(black_mask, dtype=bool)
            content_mask[rmin:rmax+1, cmin:cmax+1] = True
            frame_mask = black_mask & ~content_mask
        
        logger.info(f"Black frame detection - Has frame: {has_frame} (ratio: {max_black_ratio:.2f})")
        logger.info(f"Content area: {content_area}")
        
        return has_frame, content_area, frame_mask, max_black_ratio
        
    except Exception as e:
        logger.error(f"Frame detection error: {str(e)}")
        return False, (0, 0, img.width-1, img.height-1), None, 0.0

def remove_black_frame_replicate(img, frame_mask):
    """Remove black frame using Replicate FLUX Fill"""
    if not REPLICATE_AVAILABLE or not REQUESTS_AVAILABLE:
        logger.warning("Replicate or Requests not available")
        return img
    
    try:
        mask_img = Image.fromarray((frame_mask * 255).astype(np.uint8), mode='L')
        
        def img_to_base64(image):
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            return img_base64.rstrip('=')  # Remove padding
        
        img_base64 = img_to_base64(img)
        mask_base64 = img_to_base64(mask_img)
        
        logger.info("Starting Replicate FLUX Fill inpainting...")
        
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
    """Simple frame removal by cropping"""
    try:
        cmin, rmin, cmax, rmax = content_area
        
        padding = 10
        cmin = max(0, cmin - padding)
        rmin = max(0, rmin - padding)
        cmax = min(img.width, cmax + padding)
        rmax = min(img.height, rmax + padding)
        
        cropped = img.crop((cmin, rmin, cmax, rmax))
        
        new_img = Image.new('RGB', img.size, (250, 250, 250))
        x_offset = (img.width - cropped.width) // 2
        y_offset = (img.height - cropped.height) // 2
        new_img.paste(cropped, (x_offset, y_offset))
        
        logger.info(f"Simple frame removal: cropped {cmax-cmin}x{rmax-rmin}")
        return new_img
        
    except Exception as e:
        logger.error(f"Simple frame removal error: {str(e)}")
        return img

def apply_wedding_ring_enhancement(img, metal_type):
    """Apply wedding ring enhancement"""
    try:
        brightness_enhancer = ImageEnhance.Brightness(img)
        img = brightness_enhancer.enhance(1.05)
        
        color_enhancer = ImageEnhance.Color(img)
        img = color_enhancer.enhance(0.95)
        
        if metal_type == "yellow_gold" and NUMPY_AVAILABLE:
            img_array = np.array(img)
            img_array[:, :, 0] = np.minimum(255, img_array[:, :, 0] * 1.08).astype(np.uint8)
            img_array[:, :, 1] = np.minimum(255, img_array[:, :, 1] * 1.05).astype(np.uint8)
            img = Image.fromarray(img_array)
            
        elif metal_type == "rose_gold" and NUMPY_AVAILABLE:
            img_array = np.array(img)
            img_array[:, :, 0] = np.minimum(255, img_array[:, :, 0] * 1.10).astype(np.uint8)
            img_array[:, :, 2] = np.minimum(255, img_array[:, :, 2] * 1.03).astype(np.uint8)
            img = Image.fromarray(img_array)
            
        elif metal_type == "white_gold":
            contrast_enhancer = ImageEnhance.Contrast(img)
            img = contrast_enhancer.enhance(1.08)
            
        elif metal_type == "plain_white":
            brightness_enhancer = ImageEnhance.Brightness(img)
            img = brightness_enhancer.enhance(1.08)
        
        if NUMPY_AVAILABLE:
            img_array = np.array(img)
            mask = np.all(img_array > 200, axis=-1)
            if mask.any():
                for c in range(3):
                    img_array[mask, c] = np.minimum(255, img_array[mask, c] * 1.05).astype(np.uint8)
                img = Image.fromarray(img_array)
        
        sharpness_enhancer = ImageEnhance.Sharpness(img)
        img = sharpness_enhancer.enhance(1.15)
        
        logger.info(f"Wedding ring enhancement applied for {metal_type}")
        return img
        
    except Exception as e:
        logger.error(f"Enhancement error: {str(e)}")
        return img

def crop_to_ring_area(img, padding_ratio=0.05):
    """Crop to wedding ring area with 90% fill"""
    try:
        if not NUMPY_AVAILABLE:
            return img
            
        img_array = np.array(img)
        gray = np.mean(img_array, axis=2)
        
        non_bg = gray < 200
        rows = np.any(non_bg, axis=1)
        cols = np.any(non_bg, axis=0)
        
        if rows.any() and cols.any():
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            height = rmax - rmin
            width = cmax - cmin
            padding_h = int(height * padding_ratio)
            padding_w = int(width * padding_ratio)
            
            rmin = max(0, rmin - padding_h)
            rmax = min(img_array.shape[0], rmax + padding_h)
            cmin = max(0, cmin - padding_w)
            cmax = min(img_array.shape[1], cmax + padding_w)
            
            cropped = img.crop((cmin, rmin, cmax, rmax))
            logger.info(f"Cropped to ring area: {cmax-cmin}x{rmax-rmin}")
            return cropped
        else:
            logger.warning("No ring area detected")
            return img
            
    except Exception as e:
        logger.error(f"Ring crop error: {str(e)}")
        return img

def resize_to_thumbnail(img, target_width=1000, target_height=1300):
    """Resize to 1000x1300 thumbnail"""
    try:
        img_width, img_height = img.size
        width_ratio = target_width / img_width
        height_ratio = target_height / img_height
        
        ratio = min(width_ratio, height_ratio) * 0.9
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        
        resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        final_img = Image.new('RGB', (target_width, target_height), (250, 250, 250))
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        final_img.paste(resized, (x_offset, y_offset))
        
        logger.info(f"Final thumbnail created: {target_width}x{target_height}")
        return final_img
        
    except Exception as e:
        logger.error(f"Thumbnail resize error: {str(e)}")
        return img

def image_to_base64(img):
    """Convert PIL Image to base64 - REMOVE PADDING for Make.com"""
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
    """RunPod handler for thumbnail generation V10"""
    start_time = time.time()
    
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"Thumbnail Handler {VERSION} - Safe JSON & NumPy Fix")
        logger.info(f"{'='*60}")
        
        job_input = job.get('input', {})
        logger.info(f"Input keys: {list(job_input.keys())}")
        
        if job_input.get('debug_mode', False):
            return {
                "output": {
                    "status": "debug_success",
                    "message": f"{VERSION} handler working - Safe JSON conversion",
                    "version": VERSION,
                    "features": [
                        "Safe JSON conversion (no np.bool)",
                        "Black frame detection & removal",
                        "Metal type detection (4 types)",
                        "Wedding ring enhancement",
                        "1000x1300 thumbnail generation",
                        "Make.com compatible"
                    ]
                }
            }
        
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
        
        img = decode_base64_image(image_data_str)
        original_size = img.size
        logger.info(f"Original image size: {original_size}")
        
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        
        metal_type = detect_metal_type(img)
        has_frame, content_area, frame_mask, frame_ratio = detect_black_frame(img)
        
        if has_frame and frame_mask is not None:
            if REPLICATE_AVAILABLE:
                logger.info("Removing black frame with Replicate FLUX Fill")
                img = remove_black_frame_replicate(img, frame_mask)
            else:
                logger.info("Removing black frame with simple crop")
                img = simple_frame_removal(img, content_area)
        else:
            logger.info("No black frame detected")
        
        img = apply_wedding_ring_enhancement(img, metal_type)
        img = crop_to_ring_area(img)
        thumbnail = resize_to_thumbnail(img)
        thumbnail_base64 = image_to_base64(thumbnail)
        
        # Use safe JSON conversion
        processing_info = {
            "original_size": [original_size[0], original_size[1]],
            "metal_type": str(metal_type),
            "has_black_frame": bool(has_frame),
            "frame_ratio": float(frame_ratio),
            "content_area": [int(content_area[0]), int(content_area[1]), int(content_area[2]), int(content_area[3])],
            "replicate_used": bool(REPLICATE_AVAILABLE and has_frame),
            "final_size": [1000, 1300],
            "enhancement_applied": True,
            "processing_time": round(time.time() - start_time, 2),
            "version": VERSION
        }
        
        result = {
            "output": {
                "thumbnail": thumbnail_base64,
                "processing_info": safe_json_convert(processing_info),
                "status": "success"
            }
        }
        
        logger.info(f"Thumbnail generation completed in {processing_info['processing_time']}s")
        logger.info(f"Metal type: {metal_type}, Frame removed: {has_frame}")
        
        return safe_json_convert(result)
        
    except Exception as e:
        error_msg = f"Handler error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        error_result = {
            "output": {
                "error": error_msg,
                "status": "error",
                "version": VERSION,
                "processing_time": round(time.time() - start_time, 2)
            }
        }
        
        return safe_json_convert(error_result)

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
