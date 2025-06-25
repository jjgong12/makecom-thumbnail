import runpod
import base64
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2
import logging
import time
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VERSION = "Thumbnail_V68_BASIC"

def decode_base64_image(base64_str):
    """Decode base64 with padding fix"""
    try:
        # Remove data URI prefix
        if 'base64,' in base64_str:
            base64_str = base64_str.split('base64,')[1]
        
        # Fix padding
        base64_str = base64_str.strip()
        padding = 4 - len(base64_str) % 4
        if padding != 4:
            base64_str += '=' * padding
        
        return base64.b64decode(base64_str)
    except Exception as e:
        logger.error(f"Base64 decode error: {str(e)}")
        raise

def apply_gamma_correction(image, gamma):
    """Apply gamma correction"""
    try:
        img_array = np.array(image).astype(float) / 255.0
        corrected = np.power(img_array, gamma)
        corrected = (corrected * 255).astype(np.uint8)
        return Image.fromarray(corrected)
    except Exception as e:
        logger.error(f"Gamma error: {str(e)}")
        return image

def detect_ring_color(image):
    """Detect ring color from center region"""
    try:
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # Get center region
        center_y, center_x = h // 2, w // 2
        region_size = min(h, w) // 3
        center_region = img_array[
            center_y - region_size:center_y + region_size,
            center_x - region_size:center_x + region_size
        ]
        
        # RGB average
        avg_r = np.mean(center_region[:, :, 0])
        avg_g = np.mean(center_region[:, :, 1])
        avg_b = np.mean(center_region[:, :, 2])
        
        # HSV for better detection
        hsv = cv2.cvtColor(center_region, cv2.COLOR_RGB2HSV)
        avg_h = np.mean(hsv[:, :, 0])
        avg_s = np.mean(hsv[:, :, 1])
        avg_v = np.mean(hsv[:, :, 2])
        
        logger.info(f"RGB: R={avg_r:.1f}, G={avg_g:.1f}, B={avg_b:.1f}")
        logger.info(f"HSV: H={avg_h:.1f}, S={avg_s:.1f}, V={avg_v:.1f}")
        
        # Color detection logic
        if avg_r > avg_b + 20 and avg_g > avg_b + 10:
            return 'yellow_gold'
        elif avg_r > avg_g + 5 and avg_r > avg_b + 10:
            return 'rose_gold'
        elif avg_v > 200 and avg_s < 25:
            return 'white'
        else:
            return 'white_gold'
            
    except Exception as e:
        logger.error(f"Color detection error: {str(e)}")
        return 'yellow_gold'

def enhance_by_color(image, color):
    """Apply color-specific enhancement"""
    try:
        if color == 'yellow_gold':
            # Warm enhancement
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.1)
            
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(1.1)
            
            image = apply_gamma_correction(image, 0.9)
            
        elif color == 'rose_gold':
            # Soft enhancement
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.08)
            
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(1.05)
            
            image = apply_gamma_correction(image, 0.85)
            
        elif color == 'white':
            # Strong white enhancement
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.35)
            
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(0.5)
            
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.15)
            
            image = apply_gamma_correction(image, 0.7)
            
        else:  # white_gold
            # Cool metallic
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.15)
            
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(0.8)
            
            image = apply_gamma_correction(image, 0.8)
        
        # Sharpen all
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.5)
        
        return image
        
    except Exception as e:
        logger.error(f"Enhancement error: {str(e)}")
        return image

def create_thumbnail(image, target_size=(1000, 1300)):
    """Create centered thumbnail with 55% crop"""
    try:
        img_width, img_height = image.size
        target_ratio = target_size[0] / target_size[1]
        
        # Use 55% of image
        crop_percentage = 0.55
        
        # Calculate crop size
        if img_width / img_height > target_ratio:
            crop_height = int(img_height * crop_percentage)
            crop_width = int(crop_height * target_ratio)
        else:
            crop_width = int(img_width * crop_percentage)
            crop_height = int(crop_width / target_ratio)
        
        # Center crop
        left = (img_width - crop_width) // 2
        top = (img_height - crop_height) // 2
        
        # Crop and resize
        cropped = image.crop((left, top, left + crop_width, top + crop_height))
        thumbnail = cropped.resize(target_size, Image.Resampling.LANCZOS)
        
        return thumbnail
        
    except Exception as e:
        logger.error(f"Crop error: {str(e)}")
        return image.resize(target_size, Image.Resampling.LANCZOS)

def process_thumbnail(job):
    """Main processing function"""
    start_time = time.time()
    
    try:
        # Extract input
        job_input = job.get('input', {})
        
        # Find image data
        image_data = None
        if isinstance(job_input, dict):
            for key in ['image', 'image_base64', 'base64_image', 'enhanced_image']:
                if key in job_input:
                    image_data = job_input[key]
                    break
        elif isinstance(job_input, str):
            image_data = job_input
        
        if not image_data:
            return {
                "output": {
                    "error": "No image provided",
                    "status": "error"
                }
            }
        
        # Process image
        logger.info("Processing image...")
        
        # Decode and open
        image_bytes = decode_base64_image(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        logger.info(f"Original size: {image.size}")
        
        # Detect color
        detected_color = detect_ring_color(image)
        logger.info(f"Detected color: {detected_color}")
        
        # Apply enhancement
        enhanced = enhance_by_color(image, detected_color)
        
        # Create thumbnail
        thumbnail = create_thumbnail(enhanced, (1000, 1300))
        
        # Convert to base64
        output_buffer = BytesIO()
        thumbnail.save(output_buffer, format="PNG", optimize=True)
        final_base64 = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
        
        # Prepare results
        result_no_padding = final_base64.rstrip('=')
        result_with_padding = final_base64
        
        # Color map
        color_map = {
            'yellow_gold': '#FFD700',
            'rose_gold': '#E8B4B8',
            'white_gold': '#F5F5F5',
            'white': '#FFFFFF'
        }
        
        return {
            "output": {
                "thumbnail": result_no_padding,
                "thumbnail_with_padding": result_with_padding,
                "color": color_map.get(detected_color, '#FFD700'),
                "status": "success",
                "message": f"Thumbnail created with v68 basic processing",
                "processing_time": f"{time.time() - start_time:.2f}s",
                "detected_color": detected_color,
                "final_size": [1000, 1300],
                "settings": {
                    "size": "1000x1300",
                    "zoom_level": "55%",
                    "enhancement": "v68_basic",
                    "color_name": detected_color,
                    "color_hex": color_map.get(detected_color, '#FFD700')
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        return {
            "output": {
                "error": str(e),
                "status": "error"
            }
        }

# RunPod handler
logger.info(f"Starting RunPod {VERSION}...")
runpod.serverless.start({"handler": process_thumbnail})
