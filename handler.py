import runpod
import os
import json
import time
import base64
import numpy as np
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VERSION = "V71-BrightClean"

def find_input_data(data):
    """Find input data recursively"""
    if isinstance(data, dict):
        if any(key in data for key in ['image', 'url', 'image_url', 'imageUrl', 'image_base64', 'imageBase64']):
            return data
        
        if 'input' in data:
            return find_input_data(data['input'])
        
        for key in ['job', 'payload', 'data']:
            if key in data:
                result = find_input_data(data[key])
                if result:
                    return result
    return data

def download_image_from_url(url):
    """Download image from URL"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8'
    }
    
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))

def base64_to_image(base64_string):
    """Convert base64 to PIL Image"""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    padding = 4 - len(base64_string) % 4
    if padding != 4:
        base64_string += '=' * padding
    
    img_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(img_data))

def apply_basic_enhancement(image):
    """Apply basic enhancement matching Enhancement handler"""
    if image.mode != 'RGB':
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        else:
            image = image.convert('RGB')
    
    # Match Enhancement V71 values
    brightness = ImageEnhance.Brightness(image)
    image = brightness.enhance(1.12)  # Default brightness
    
    contrast = ImageEnhance.Contrast(image)
    image = contrast.enhance(1.08)
    
    color = ImageEnhance.Color(image)
    image = color.enhance(1.05)
    
    return image

def detect_ring_color(image):
    """Detect ring color - same as Enhancement"""
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    
    # Center 50% region
    center_y, center_x = height // 2, width // 2
    crop_size = min(height, width) // 2
    
    y1 = max(0, center_y - crop_size // 2)
    y2 = min(height, center_y + crop_size // 2)
    x1 = max(0, center_x - crop_size // 2)
    x2 = min(width, center_x + crop_size // 2)
    
    center_region = img_array[y1:y2, x1:x2]
    hsv = cv2.cvtColor(center_region, cv2.COLOR_RGB2HSV)
    
    avg_saturation = np.mean(hsv[:, :, 1])
    avg_value = np.mean(hsv[:, :, 2])
    
    # RGB analysis
    r_mean = np.mean(center_region[:, :, 0])
    g_mean = np.mean(center_region[:, :, 1])
    b_mean = np.mean(center_region[:, :, 2])
    
    max_rgb = max(r_mean, g_mean, b_mean)
    if max_rgb > 0:
        r_norm = r_mean / max_rgb
        g_norm = g_mean / max_rgb
        b_norm = b_mean / max_rgb
    else:
        r_norm = g_norm = b_norm = 1.0
    
    # Color detection
    if avg_saturation < 25:
        color = "무도금화이트" if avg_value > 200 else "화이트골드"
    elif r_norm > 0.95 and 0.85 < g_norm < 0.95:
        color = "로즈골드" if avg_saturation > 40 else "옐로우골드"
    elif abs(r_norm - g_norm) < 0.1 and abs(g_norm - b_norm) < 0.1:
        color = "화이트골드"
    else:
        warmth = (r_norm + g_norm) / 2 - b_norm
        color = "옐로우골드" if warmth > 0.1 else "화이트골드"
    
    logger.info(f"Detected color: {color}")
    return color

def apply_color_specific_enhancement(image, color):
    """Apply lighter color-specific enhancement to match Enhancement handler"""
    enhanced = image.copy()
    
    if color == '무도금화이트':
        # Lighter pure white enhancement
        brightness = ImageEnhance.Brightness(enhanced)
        enhanced = brightness.enhance(1.08)  # Much lighter than before
        
        # Gentle desaturation
        color_enhancer = ImageEnhance.Color(enhanced)
        enhanced = color_enhancer.enhance(0.8)
        
        # Very slight blue tint
        img_array = np.array(enhanced)
        img_array[:, :, 2] = np.minimum(img_array[:, :, 2] * 1.03, 255).astype(np.uint8)
        enhanced = Image.fromarray(img_array)
    
    elif color == '옐로우골드':
        # Light warm enhancement
        brightness = ImageEnhance.Brightness(enhanced)
        enhanced = brightness.enhance(1.05)
        
        img_array = np.array(enhanced)
        img_array[:, :, 0] = np.minimum(img_array[:, :, 0] * 1.03, 255).astype(np.uint8)
        img_array[:, :, 1] = np.minimum(img_array[:, :, 1] * 1.02, 255).astype(np.uint8)
        enhanced = Image.fromarray(img_array)
    
    elif color == '로즈골드':
        # Light rose enhancement
        brightness = ImageEnhance.Brightness(enhanced)
        enhanced = brightness.enhance(1.06)
        
        img_array = np.array(enhanced)
        img_array[:, :, 0] = np.minimum(img_array[:, :, 0] * 1.04, 255).astype(np.uint8)
        enhanced = Image.fromarray(img_array)
    
    elif color == '화이트골드':
        # Light cool enhancement
        brightness = ImageEnhance.Brightness(enhanced)
        enhanced = brightness.enhance(1.08)
        
        contrast = ImageEnhance.Contrast(enhanced)
        enhanced = contrast.enhance(1.05)
        
        img_array = np.array(enhanced)
        img_array[:, :, 2] = np.minimum(img_array[:, :, 2] * 1.02, 255).astype(np.uint8)
        enhanced = Image.fromarray(img_array)
    
    # Light sharpening
    sharpness = ImageEnhance.Sharpness(enhanced)
    enhanced = sharpness.enhance(1.3)
    
    return enhanced

def create_thumbnail_with_crop(image, size=(1000, 1300)):
    """Create thumbnail with minimal 10% crop"""
    logger.info(f"Creating thumbnail - original size: {image.size}")
    
    target_ratio = 1000 / 1300  # 0.769
    img_width, img_height = image.size
    
    # Only 10% crop for minimal zoom
    crop_percentage = 0.1
    
    # Calculate crop area
    crop_width = int(img_width * (1 - crop_percentage))
    crop_height = int(img_height * (1 - crop_percentage))
    
    # Adjust for target ratio
    current_ratio = crop_width / crop_height
    
    if current_ratio > target_ratio:
        # Too wide - increase height
        crop_height = int(crop_width / target_ratio)
    else:
        # Too tall - increase width
        crop_width = int(crop_height * target_ratio)
    
    # Ensure we don't exceed image bounds
    crop_width = min(crop_width, img_width)
    crop_height = min(crop_height, img_height)
    
    # Center align
    x_offset = max(0, (img_width - crop_width) // 2)
    y_offset = max(0, (img_height - crop_height) // 2)
    
    # Crop
    crop_box = (x_offset, y_offset, x_offset + crop_width, y_offset + crop_height)
    cropped = image.crop(crop_box)
    
    # Resize to final size
    thumbnail = cropped.resize(size, Image.Resampling.LANCZOS)
    
    logger.info(f"Thumbnail created: {thumbnail.size}")
    return thumbnail

def apply_center_vignette(image):
    """Apply very subtle center vignette"""
    width, height = image.size
    
    # Create gradient mask
    mask = Image.new('L', (width, height), 255)
    mask_array = np.array(mask)
    
    center_x, center_y = width // 2, height // 2
    
    for y in range(height):
        for x in range(width):
            dx = (x - center_x) / center_x
            dy = (y - center_y) / center_y
            dist = np.sqrt(dx**2 + dy**2)
            
            # Very subtle vignette - almost imperceptible
            brightness = int(255 * (1 - dist * 0.05))
            brightness = max(242, brightness)  # Very bright minimum
            mask_array[y, x] = brightness
    
    mask = Image.fromarray(mask_array)
    
    # Apply vignette with very light background
    vignette_layer = Image.new('RGB', (width, height), (250, 250, 250))
    return Image.composite(image, vignette_layer, mask)

def image_to_base64(image, format='JPEG'):
    """Convert PIL Image to base64"""
    buffered = BytesIO()
    if format == 'JPEG' and image.mode == 'RGBA':
        rgb_image = Image.new('RGB', image.size, (255, 255, 255))
        rgb_image.paste(image, mask=image.split()[3])
        image = rgb_image
    
    image.save(buffered, format=format, quality=95)
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    # Remove padding for Make.com
    return img_base64.rstrip('=')

def handler(event):
    """Thumbnail handler"""
    try:
        logger.info(f"[{VERSION}] Event received")
        
        # Find input data
        input_data = find_input_data(event)
        
        if not input_data:
            raise ValueError("No input data found")
        
        # Load image
        image = None
        
        if isinstance(input_data, dict):
            if 'image_base64' in input_data or 'imageBase64' in input_data:
                base64_str = input_data.get('image_base64') or input_data.get('imageBase64')
                image = base64_to_image(base64_str)
            elif 'url' in input_data or 'image_url' in input_data or 'imageUrl' in input_data:
                image_url = input_data.get('url') or input_data.get('image_url') or input_data.get('imageUrl')
                image = download_image_from_url(image_url)
        elif isinstance(input_data, str):
            if input_data.startswith('http'):
                image = download_image_from_url(input_data)
            else:
                image = base64_to_image(input_data)
        
        if not image:
            raise ValueError("Could not load image")
        
        logger.info(f"Image loaded: {image.size}")
        
        # 1. Apply basic enhancement (matching Enhancement handler)
        enhanced_image = apply_basic_enhancement(image)
        
        # 2. Create thumbnail with minimal crop (10%)
        thumbnail = create_thumbnail_with_crop(enhanced_image)
        
        # 3. Detect color
        detected_color = detect_ring_color(thumbnail)
        
        # 4. Apply light color-specific enhancement
        thumbnail = apply_color_specific_enhancement(thumbnail, detected_color)
        
        # 5. Apply very subtle vignette
        thumbnail = apply_center_vignette(thumbnail)
        
        # 6. Final brightness boost for clean look
        brightness = ImageEnhance.Brightness(thumbnail)
        thumbnail = brightness.enhance(1.02)
        
        # Convert to base64
        thumbnail_base64 = image_to_base64(thumbnail)
        
        # Return result
        return {
            "output": {
                "thumbnail": thumbnail_base64,
                "size": list(thumbnail.size),
                "detected_color": detected_color,
                "format": "base64_no_padding",
                "version": VERSION
            }
        }
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {
            "output": {
                "error": str(e),
                "status": "failed",
                "version": VERSION
            }
        }

runpod.serverless.start({"handler": handler})
