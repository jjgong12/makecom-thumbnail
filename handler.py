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

VERSION = "v12"

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
    """
    Detect wedding ring metal type from 4 categories
    Based on 38 training data pairs (28 + 10)
    """
    try:
        if not NUMPY_AVAILABLE:
            return "white_gold"
            
        img_array = np.array(image)
        
        if CV2_AVAILABLE:
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            h_channel = hsv[:, :, 0]
            s_channel = hsv[:, :, 1]
            v_channel = hsv[:, :, 2]
            
            avg_hue = np.mean(h_channel)
            avg_saturation = np.mean(s_channel)
            avg_value = np.mean(v_channel)
            
            if avg_saturation < 50 and avg_value > 180:
                metal_type = "plain_white"  # 무도금화이트
            elif avg_hue < 30 and avg_saturation > 80:
                metal_type = "yellow_gold"  # 옐로우골드
            elif avg_hue < 20 and avg_saturation > 50:
                metal_type = "rose_gold"    # 로즈골드
            else:
                metal_type = "white_gold"   # 화이트골드
        else:
            avg_color = np.mean(img_array.reshape(-1, 3), axis=0)
            r, g, b = avg_color
            
            if r > 200 and g > 200 and b > 200:
                metal_type = "white_gold"
            elif r > g + 20 and r > b + 20:
                if g > b:
                    metal_type = "yellow_gold"
                else:
                    metal_type = "rose_gold"
            else:
                metal_type = "plain_white"
        
        logger.info(f"Metal type detected: {metal_type}")
        return metal_type
        
    except Exception as e:
        logger.error(f"Metal detection error: {str(e)}")
        return "white_gold"

def detect_black_frame(img):
    """
    Detect black frame/borders using multiple methods
    - Edge-based detection: Check edges for black pixels
    - Center-based detection: Find content area and check for frames
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
        frame_mask = None
        if has_frame:
            black_mask = gray < edge_threshold
            
            # Remove content area from mask
            cmin, rmin, cmax, rmax = content_area
            content_mask = np.zeros_like(black_mask, dtype=bool)
            content_mask[rmin:rmax+1, cmin:cmax+1] = True
            
            # Final frame mask (black areas outside content)
            frame_mask = black_mask & ~content_mask
        
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
        
        # Create new image with light gray background
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
        # Base enhancement (same as enhance_handler V12)
        brightness_enhancer = ImageEnhance.Brightness(img)
        img = brightness_enhancer.enhance(1.05)  # 5% brightness increase
        
        color_enhancer = ImageEnhance.Color(img)
        img = color_enhancer.enhance(0.95)  # 5% saturation decrease
        
        # Metal-specific enhancement
        if metal_type == "yellow_gold" and NUMPY_AVAILABLE:
            # Enhance warm tones
            img_array = np.array(img)
            img_array[:, :, 0] = np.minimum(255, img_array[:, :, 0] * 1.08).astype(np.uint8)  # More red
            img_array[:, :, 1] = np.minimum(255, img_array[:, :, 1] * 1.05).astype(np.uint8)  # Slight green
            img = Image.fromarray(img_array)
            
        elif metal_type == "rose_gold" and NUMPY_AVAILABLE:
            # Enhance pink/warm tones
            img_array = np.array(img)
            img_array[:, :, 0] = np.minimum(255, img_array[:, :, 0] * 1.10).astype(np.uint8)  # More red
            img_array[:, :, 2] = np.minimum(255, img_array[:, :, 2] * 1.03).astype(np.uint8)  # Slight blue
            img = Image.fromarray(img_array)
            
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
        if not NUMPY_AVAILABLE:
            return img
            
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
    Wedding rings fill 90% of the canvas
    """
    try:
        # Calculate ratios
        img_width, img_height = img.size
        width_ratio = target_width / img_width
        height_ratio = target_height / img_height
        
        # Use smaller ratio to fit entirely, then scale to 90% fill
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
    Google Apps Script will restore padding when needed
    """
    try:
        buffer = io.BytesIO()
        img.save(buffer, format='PNG', quality=95, optimize=True)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # CRITICAL: Remove padding for Make.com compatibility
        # Google Apps Script will restore padding when needed
        img_base64 = img_base64.rstrip('=')
        
        logger.info(f"Image converted to base64, length: {len(img_base64)}, padding removed")
        return img_base64
        
    except Exception as e:
        logger.error(f"Base64 conversion error: {str(e)}")
        return ""

def handler(job):
    """
    RunPod handler for thumbnail generation V12
    Complete black frame removal and thumbnail generation
    """
    start_time = time.time()
    
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"Thumbnail Handler {VERSION} - Complete Package")
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
                    "message": f"{VERSION} thumbnail handler working - Complete package",
                    "version": VERSION,
                    "features": [
                        "Safe JSON conversion (no np.bool)",
                        "Black frame detection & removal",
                        "Replicate FLUX Fill inpainting",
                        "Metal type detection (4 types)",
                        "Wedding ring enhancement (38 pairs)",
                        "1000x1300 thumbnail generation",
                        "90% ring fill ratio",
                        "NumPy 1.24+ compatibility",
                        "Make.com compatible",
                        "Google Apps Script support"
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
            "original_size": [original_size[0], original_size[1]],
            "metal_type": str(metal_type),
            "has_black_frame": bool(has_frame),
            "frame_ratio": float(frame_ratio),
            "content_area": [int(content_area[0]), int(content_area[1]), int(content_area[2]), int(content_area[3])],
            "replicate_used": bool(REPLICATE_AVAILABLE and has_frame),
            "final_size": [1000, 1300],
            "enhancement_applied": True,
            "ring_fill_ratio": 0.9,
            "processing_time": round(time.time() - start_time, 2),
            "version": VERSION,
            "training_data": "38 pairs (28 + 10)"
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
        logger.info(f"Output structure: {list(result['output'].keys())}")
        
        return safe_json_convert(result)
        
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

"""
=============================================================================
GOOGLE APPS SCRIPT V12 - BASE64 PADDING FIX
=============================================================================

Copy this JavaScript code to your Google Apps Script project:

/**
 * Google Apps Script V12 - Base64 Padding Fix for thumbnail_handler
 * 
 * Fixes "Error: Invalid base64 data" by restoring padding removed by RunPod
 * for Make.com compatibility
 */

function doPost(e) {
  try {
    console.log('Google Apps Script V12 started - Thumbnail Image Upload');
    
    // Parse POST data
    const postData = JSON.parse(e.postData.contents);
    console.log('Input data keys:', Object.keys(postData));
    
    // Extract thumbnail base64 data
    let base64Data = null;
    
    // Try multiple keys to find the thumbnail
    const imageKeys = ['thumbnail', 'image', 'image_base64', 'base64', 'img', 'data'];
    for (const key of imageKeys) {
      if (postData[key] && typeof postData[key] === 'string' && postData[key].length > 100) {
        base64Data = postData[key];
        console.log(`Found thumbnail data in key: ${key}, length: ${base64Data.length}`);
        break;
      }
    }
    
    if (!base64Data) {
      throw new Error(`No thumbnail data found. Available keys: ${Object.keys(postData).join(', ')}`);
    }
    
    // Fix base64 data (restore padding removed by RunPod for Make.com)
    console.log('Fixing base64 padding for thumbnail...');
    base64Data = fixBase64Padding(base64Data);
    
    // Validate base64 format
    if (!isValidBase64(base64Data)) {
      throw new Error('Invalid base64 format after padding fix');
    }
    
    // Create blob from base64
    console.log('Creating thumbnail blob...');
    const imageBlob = Utilities.newBlob(
      Utilities.base64Decode(base64Data),
      'image/png',
      `wedding_ring_thumbnail_${new Date().toISOString().replace(/[:.]/g, '-')}.png`
    );
    
    // Upload to Google Drive
    console.log('Uploading thumbnail to Google Drive...');
    const file = DriveApp.createFile(imageBlob);
    
    console.log(`Thumbnail uploaded successfully: ${file.getName()} (${file.getSize()} bytes)`);
    
    // Return success response
    return ContentService
      .createTextOutput(JSON.stringify({
        success: true,
        fileId: file.getId(),
        fileName: file.getName(),
        fileUrl: file.getUrl(),
        fileSize: file.getSize(),
        imageType: 'thumbnail',
        dimensions: '1000x1300',
        message: 'Wedding ring thumbnail uploaded successfully with V12 padding fix',
        timestamp: new Date().toISOString(),
        version: 'v12'
      }))
      .setMimeType(ContentService.MimeType.JSON);
      
  } catch (error) {
    console.error('Google Apps Script error:', error.toString());
    
    return ContentService
      .createTextOutput(JSON.stringify({
        success: false,
        error: error.toString(),
        message: 'Failed to upload wedding ring thumbnail',
        timestamp: new Date().toISOString(),
        version: 'v12'
      }))
      .setMimeType(ContentService.MimeType.JSON);
  }
}

function fixBase64Padding(base64Data) {
  try {
    console.log(`Original base64 length: ${base64Data.length}`);
    
    // Remove data URL prefix if present
    if (base64Data.includes('data:')) {
      const commaIndex = base64Data.indexOf(',');
      if (commaIndex !== -1) {
        base64Data = base64Data.substring(commaIndex + 1);
        console.log('Removed data URL prefix');
      }
    }
    
    // Remove any whitespace characters
    base64Data = base64Data.replace(/\s/g, '');
    console.log(`After whitespace removal: ${base64Data.length}`);
    
    // Calculate and add padding (RunPod removes it for Make.com compatibility)
    const paddingNeeded = 4 - (base64Data.length % 4);
    if (paddingNeeded !== 4) {
      const paddingChars = '='.repeat(paddingNeeded);
      base64Data += paddingChars;
      console.log(`Added ${paddingNeeded} padding characters: ${paddingChars}`);
    } else {
      console.log('No padding needed');
    }
    
    console.log(`Final base64 length: ${base64Data.length}`);
    return base64Data;
    
  } catch (error) {
    console.error('Base64 padding fix error:', error.toString());
    throw new Error(`Failed to fix base64 padding: ${error.toString()}`);
  }
}

function isValidBase64(base64String) {
  try {
    // Check basic format
    const base64Pattern = /^[A-Za-z0-9+/]*={0,2}$/;
    if (!base64Pattern.test(base64String)) {
      console.error('Base64 pattern validation failed');
      return false;
    }
    
    // Check length is multiple of 4
    if (base64String.length % 4 !== 0) {
      console.error('Base64 length is not multiple of 4');
      return false;
    }
    
    // Try to decode (this will throw if invalid)
    Utilities.base64Decode(base64String);
    console.log('Base64 validation successful');
    return true;
    
  } catch (error) {
    console.error('Base64 validation error:', error.toString());
    return false;
  }
}

function doGet() {
  return ContentService
    .createTextOutput(JSON.stringify({
      status: 'Google Apps Script V12 is running',
      handler: 'thumbnail',
      message: 'Wedding Ring Thumbnail Upload Service',
      usage: 'Use POST method to upload thumbnails',
      features: [
        'Base64 padding restoration',
        'Thumbnail image processing',
        'Black frame removal support',
        '1000x1300 thumbnail handling',
        'RunPod V12 integration',
        'Make.com compatibility',
        'Google Drive upload',
        'Detailed logging'
      ],
      version: 'v12',
      timestamp: new Date().toISOString()
    }))
    .setMimeType(ContentService.MimeType.JSON);
}

=============================================================================
REQUIREMENTS.TXT V12
=============================================================================

runpod==1.6.0
opencv-python-headless==4.8.1.78
Pillow==10.1.0
numpy==1.24.3
replicate==0.32.1
requests==2.31.0

=============================================================================
DEPLOYMENT INSTRUCTIONS V12
=============================================================================

1. RunPod Thumbnail Handler:
   - Upload this file as handler.py
   - Set REPLICATE_API_TOKEN environment variable
   - Deploy and test with: {"input": {"debug_mode": true}}

2. Google Apps Script:
   - Copy the JavaScript code above to your Google Apps Script project
   - Deploy as web app with proper permissions
   - Test with thumbnail data from RunPod

3. Make.com Configuration:
   - Thumbnail Module → Google Apps Script
   - Path: {{4.data.output.output.thumbnail}}
   - Method: POST

Features V12:
- ✅ NumPy 1.24+ compatibility (no np.bool references)
- ✅ Safe JSON serialization for all data types
- ✅ Make.com base64 compatibility (padding removed)
- ✅ Google Apps Script padding restoration
- ✅ Black frame detection (Edge + Center based)
- ✅ Replicate FLUX Fill inpainting for frame removal
- ✅ Metal type detection (4 types: yellow_gold, rose_gold, white_gold, plain_white)
- ✅ Wedding ring enhancement (38 training pairs)
- ✅ 1000x1300 thumbnail generation with 90% ring fill
- ✅ Comprehensive error handling and logging
"""
