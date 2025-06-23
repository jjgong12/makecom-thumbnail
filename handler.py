"""
Wedding Ring Thumbnail Handler V13 - Complete Package
- Black frame detection and removal (enhanced)
- Replicate FLUX Fill API for inpainting
- NumPy 1.24+ compatibility
- Safe JSON serialization
- Make.com base64 compatibility
- 1000x1300 thumbnail with 90% ring fill
"""

import runpod
import base64
import io
import json
import time
import traceback
from typing import Dict, Any, Union, Optional, Tuple
import logging
import os

# Optional imports
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import replicate
    REPLICATE_AVAILABLE = True
except ImportError:
    REPLICATE_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VERSION = "V13"
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")

# Safe JSON conversion (same as enhance handler)
def safe_json_convert(obj):
    """Safely convert numpy/special types to JSON-serializable format"""
    if obj is None:
        return None
    
    if NUMPY_AVAILABLE:
        obj_type_str = str(type(obj))
        
        if 'numpy.bool' in obj_type_str or obj_type_str == "<class 'numpy.bool_'>":
            return bool(obj)
        
        if 'numpy.int' in obj_type_str or 'numpy.uint' in obj_type_str:
            return int(obj)
        
        if 'numpy.float' in obj_type_str:
            return float(obj)
        
        if hasattr(obj, 'tolist') and callable(getattr(obj, 'tolist')):
            return obj.tolist()
    
    if isinstance(obj, bool):
        return obj
    
    if isinstance(obj, (int, float, str)):
        return obj
    
    if isinstance(obj, list):
        return [safe_json_convert(item) for item in obj]
    
    if isinstance(obj, dict):
        return {key: safe_json_convert(value) for key, value in obj.items()}
    
    if isinstance(obj, bytes):
        return base64.b64encode(obj).decode('utf-8')
    
    return str(obj)

def find_image_data(job_input, depth=0, max_depth=5):
    """Find image data in nested structure"""
    if depth > max_depth:
        return None
    
    image_keys = ['image', 'image_base64', 'base64', 'img', 'data', 
                  'imageData', 'image_data', 'input_image', 'file']
    
    if isinstance(job_input, dict):
        for key in image_keys:
            if key in job_input and job_input[key]:
                value = job_input[key]
                if isinstance(value, str) and len(value) > 100:
                    logger.info(f"Found image in key: {key}")
                    return value
        
        for key, value in job_input.items():
            result = find_image_data(value, depth + 1, max_depth)
            if result:
                return result
    
    elif isinstance(job_input, list):
        for item in job_input:
            result = find_image_data(item, depth + 1, max_depth)
            if result:
                return result
    
    elif isinstance(job_input, str) and len(job_input) > 100:
        if job_input.startswith('data:image') or all(
            c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' 
            for c in job_input[:100]):
            return job_input
    
    return None

def decode_base64_image(base64_str):
    """Decode base64 string to PIL Image"""
    try:
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        
        base64_str = base64_str.strip()
        
        for padding in ['', '=', '==', '===']:
            try:
                padded = base64_str + padding
                img_data = base64.b64decode(padded)
                return Image.open(io.BytesIO(img_data))
            except:
                continue
        
        raise ValueError("Failed to decode base64")
        
    except Exception as e:
        logger.error(f"Base64 decode error: {str(e)}")
        raise

def detect_black_frame_advanced(img):
    """
    Advanced black frame detection using multiple methods
    Returns: (has_frame, content_area, frame_mask, frame_ratio)
    """
    if not NUMPY_AVAILABLE:
        return False, (0, 0, img.width-1, img.height-1), None, 0.0
    
    try:
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if CV2_AVAILABLE else np.mean(img_array, axis=2)
        else:
            gray = img_array
        
        # Method 1: Edge detection (for images > 1500px)
        edge_threshold = 30
        edge_pixels = 100 if max(width, height) > 1500 else 50
        
        # Check edges
        top_edge = gray[:edge_pixels, :]
        bottom_edge = gray[-edge_pixels:, :]
        left_edge = gray[:, :edge_pixels]
        right_edge = gray[:, -edge_pixels:]
        
        # Calculate black pixel ratios
        top_black = np.sum(top_edge < edge_threshold) / top_edge.size
        bottom_black = np.sum(bottom_edge < edge_threshold) / bottom_edge.size
        left_black = np.sum(left_edge < edge_threshold) / left_edge.size
        right_black = np.sum(right_edge < edge_threshold) / right_edge.size
        
        max_black_ratio = max(top_black, bottom_black, left_black, right_black)
        has_edge_frame = max_black_ratio > 0.8
        
        # Method 2: Find content boundaries
        # Use multiple thresholds for better detection
        thresholds = [30, 40, 50, 60, 70, 80]
        content_masks = []
        
        for thresh in thresholds:
            mask = gray > thresh
            content_masks.append(mask)
        
        # Combine masks
        combined_mask = np.logical_or.reduce(content_masks)
        
        # Find boundaries
        rows = np.any(combined_mask, axis=1)
        cols = np.any(combined_mask, axis=0)
        
        if rows.any() and cols.any():
            row_indices = np.where(rows)[0]
            col_indices = np.where(cols)[0]
            
            rmin, rmax = row_indices[0], row_indices[-1]
            cmin, cmax = col_indices[0], col_indices[-1]
            
            content_area = (cmin, rmin, cmax, rmax)
            
            # Check if there's significant black frame
            content_width = cmax - cmin + 1
            content_height = rmax - rmin + 1
            
            frame_width_ratio = (width - content_width) / width
            frame_height_ratio = (height - content_height) / height
            
            has_content_frame = frame_width_ratio > 0.05 or frame_height_ratio > 0.05
        else:
            content_area = (0, 0, width-1, height-1)
            has_content_frame = False
        
        # Create frame mask
        has_frame = has_edge_frame or has_content_frame
        frame_mask = None
        
        if has_frame:
            # Create detailed mask
            frame_mask = np.zeros((height, width), dtype=bool)
            
            # Mark black pixels
            for thresh in [30, 40, 50]:
                black_pixels = gray < thresh
                frame_mask |= black_pixels
            
            # Exclude content area
            cmin, rmin, cmax, rmax = content_area
            frame_mask[rmin:rmax+1, cmin:cmax+1] = False
            
            # Expand mask slightly for better inpainting
            if CV2_AVAILABLE:
                kernel = np.ones((5, 5), np.uint8)
                frame_mask = cv2.dilate(frame_mask.astype(np.uint8), kernel, iterations=2).astype(bool)
        
        frame_ratio = np.sum(frame_mask) / (width * height) if frame_mask is not None else 0
        
        logger.info(f"Black frame detection - Edge: {has_edge_frame}, Content: {has_content_frame}")
        logger.info(f"Content area: ({content_area[0]}, {content_area[1]}) to ({content_area[2]}, {content_area[3]})")
        logger.info(f"Frame ratio: {frame_ratio:.2%}")
        
        return has_frame, content_area, frame_mask, frame_ratio
        
    except Exception as e:
        logger.error(f"Frame detection error: {str(e)}")
        return False, (0, 0, img.width-1, img.height-1), None, 0.0

def remove_black_frame_replicate(img, frame_mask):
    """Remove black frame using Replicate FLUX Fill API"""
    if not REPLICATE_AVAILABLE or not REPLICATE_API_TOKEN:
        logger.warning("Replicate not available")
        return simple_frame_removal(img, frame_mask)
    
    try:
        # Convert mask to image
        mask_img = Image.fromarray((frame_mask * 255).astype(np.uint8), mode='L')
        
        # Save images to buffers
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        mask_buffer = io.BytesIO()
        mask_img.save(mask_buffer, format='PNG')
        mask_buffer.seek(0)
        
        # Run Replicate model
        logger.info("Running Replicate FLUX Fill for frame removal...")
        
        output = replicate.run(
            "lucataco/flux-fill-pro",
            input={
                "image": img_buffer,
                "mask": mask_buffer,
                "prompt": "professional product photography white background, smooth seamless gradient, studio lighting, clean minimal",
                "guidance_scale": 30,
                "steps": 50,
                "strength": 0.85
            }
        )
        
        # Get result
        if output:
            if isinstance(output, list) and len(output) > 0:
                result_url = output[0]
            else:
                result_url = str(output)
            
            response = requests.get(result_url)
            result_img = Image.open(io.BytesIO(response.content))
            logger.info("Replicate inpainting successful")
            return result_img
        
        logger.warning("No output from Replicate")
        return simple_frame_removal(img, frame_mask)
        
    except Exception as e:
        logger.error(f"Replicate error: {str(e)}")
        return simple_frame_removal(img, frame_mask)

def simple_frame_removal(img, frame_mask):
    """Simple frame removal fallback"""
    if not NUMPY_AVAILABLE or frame_mask is None:
        return img
    
    try:
        img_array = np.array(img)
        
        # Get average color from edges
        h, w = frame_mask.shape
        edge_colors = []
        
        # Sample colors from non-masked edges
        for i in range(0, w, 10):
            if not frame_mask[10, i]:
                edge_colors.append(img_array[10, i])
            if not frame_mask[h-10, i]:
                edge_colors.append(img_array[h-10, i])
        
        for i in range(0, h, 10):
            if not frame_mask[i, 10]:
                edge_colors.append(img_array[i, 10])
            if not frame_mask[i, w-10]:
                edge_colors.append(img_array[i, w-10])
        
        if edge_colors:
            avg_color = np.mean(edge_colors, axis=0).astype(np.uint8)
        else:
            avg_color = np.array([250, 250, 250])  # Light background
        
        # Fill frame area
        img_array[frame_mask] = avg_color
        
        return Image.fromarray(img_array)
        
    except Exception as e:
        logger.error(f"Simple frame removal error: {str(e)}")
        return img

def find_ring_bounds(img):
    """Find wedding ring boundaries in image"""
    if not NUMPY_AVAILABLE:
        return None
    
    try:
        img_array = np.array(img)
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if CV2_AVAILABLE else np.mean(img_array, axis=2)
        else:
            gray = img_array
        
        # Find objects (rings are usually darker than background)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) if CV2_AVAILABLE else (None, gray < 200)
        
        # Find contours
        if CV2_AVAILABLE:
            contours, _ = cv2.findContours(255 - binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                return (x, y, x + w, y + h)
        else:
            # Simple method without OpenCV
            rows = np.any(binary, axis=1)
            cols = np.any(binary, axis=0)
            
            if rows.any() and cols.any():
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                return (cmin, rmin, cmax, rmax)
        
        return None
        
    except Exception as e:
        logger.error(f"Ring detection error: {str(e)}")
        return None

def create_thumbnail(img, target_size=(1000, 1300)):
    """Create thumbnail with rings filling 90% of space"""
    try:
        # Find ring bounds
        ring_bounds = find_ring_bounds(img)
        
        if ring_bounds:
            x1, y1, x2, y2 = ring_bounds
            ring_width = x2 - x1
            ring_height = y2 - y1
            
            # Add 10% padding (so rings fill 90%)
            padding_x = int(ring_width * 0.1)
            padding_y = int(ring_height * 0.1)
            
            # Crop with padding
            x1 = max(0, x1 - padding_x)
            y1 = max(0, y1 - padding_y)
            x2 = min(img.width, x2 + padding_x)
            y2 = min(img.height, y2 + padding_y)
            
            cropped = img.crop((x1, y1, x2, y2))
        else:
            # Fallback: center crop
            cropped = img
        
        # Resize to target maintaining aspect ratio
        cropped.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # Create final image with exact dimensions
        final = Image.new('RGB', target_size, (250, 250, 250))
        
        # Paste centered
        x = (target_size[0] - cropped.width) // 2
        y = (target_size[1] - cropped.height) // 2
        final.paste(cropped, (x, y))
        
        logger.info(f"Created thumbnail: {target_size}")
        return final
        
    except Exception as e:
        logger.error(f"Thumbnail creation error: {str(e)}")
        return img.resize(target_size, Image.Resampling.LANCZOS)

def apply_enhancement(img):
    """Apply same enhancement as enhance handler"""
    try:
        # Brightness
        brightness = ImageEnhance.Brightness(img)
        img = brightness.enhance(1.05)
        
        # Color
        color = ImageEnhance.Color(img)
        img = color.enhance(0.95)
        
        # Contrast
        contrast = ImageEnhance.Contrast(img)
        img = contrast.enhance(1.03)
        
        # Slight sharpening
        img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=30, threshold=0))
        
        return img
        
    except Exception as e:
        logger.error(f"Enhancement error: {str(e)}")
        return img

def image_to_base64(img):
    """Convert to base64 without padding for Make.com"""
    try:
        buffer = io.BytesIO()
        img.save(buffer, format='PNG', quality=95, optimize=True)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Remove padding for Make.com
        img_base64 = img_base64.rstrip('=')
        
        logger.info(f"Base64 length: {len(img_base64)}, padding removed")
        return img_base64
        
    except Exception as e:
        logger.error(f"Base64 conversion error: {str(e)}")
        return ""

def handler(job):
    """RunPod handler for thumbnail generation V13"""
    start_time = time.time()
    
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"Thumbnail Handler {VERSION} Started")
        logger.info(f"Features: Advanced black frame removal, FLUX Fill inpainting")
        logger.info(f"{'='*60}")
        
        # Get input
        job_input = job.get('input', {})
        
        # Debug mode
        if job_input.get('debug_mode', False):
            return {
                "output": {
                    "status": "debug_success",
                    "message": f"{VERSION} thumbnail handler working",
                    "version": VERSION,
                    "features": [
                        "Advanced black frame detection",
                        "Replicate FLUX Fill inpainting",
                        "Ring detection and centering",
                        "1000x1300 thumbnail generation",
                        "90% ring fill ratio",
                        "Same enhancement as main handler"
                    ]
                }
            }
        
        # Find image
        image_data_str = find_image_data(job_input)
        if not image_data_str:
            return {
                "output": {
                    "error": "No image found",
                    "status": "error",
                    "version": VERSION
                }
            }
        
        # Decode
        img = decode_base64_image(image_data_str)
        logger.info(f"Image decoded: {img.size}")
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Detect and remove black frame
        has_frame, content_area, frame_mask, frame_ratio = detect_black_frame_advanced(img)
        
        if has_frame and frame_ratio > 0.02:  # Only process if >2% frame
            logger.info(f"Black frame detected ({frame_ratio:.1%}), removing...")
            img = remove_black_frame_replicate(img, frame_mask)
        else:
            logger.info("No significant black frame detected")
        
        # Apply enhancements
        img = apply_enhancement(img)
        
        # Create thumbnail
        thumbnail = create_thumbnail(img, (1000, 1300))
        
        # Convert to base64
        thumbnail_base64 = image_to_base64(thumbnail)
        
        if not thumbnail_base64:
            return {
                "output": {
                    "error": "Failed to convert thumbnail",
                    "status": "error",
                    "version": VERSION
                }
            }
        
        # Response
        processing_time = time.time() - start_time
        response_data = {
            "thumbnail": thumbnail_base64,
            "has_black_frame": has_frame,
            "frame_ratio": round(frame_ratio, 3) if frame_ratio else 0,
            "processing_time": round(processing_time, 2),
            "original_size": safe_json_convert(img.size),
            "thumbnail_size": [1000, 1300],
            "version": VERSION,
            "message": "Thumbnail created successfully"
        }
        
        safe_response = safe_json_convert(response_data)
        
        logger.info(f"Thumbnail complete in {processing_time:.2f}s")
        
        return {"output": safe_response}
        
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Handler error: {str(e)}\n{error_trace}")
        
        return {
            "output": safe_json_convert({
                "error": str(e),
                "error_trace": error_trace,
                "status": "error",
                "version": VERSION
            })
        }

# RunPod handler
runpod.serverless.start({"handler": handler})

# Test mode
if __name__ == "__main__":
    print(f"Testing {VERSION} Thumbnail Handler...")
    test_job = {
        "input": {
            "debug_mode": True
        }
    }
    result = handler(test_job)
    print(json.dumps(result, indent=2))
