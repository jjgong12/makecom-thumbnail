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

VERSION = "V80-SmallRing"

def find_input_data(data):
    """Find input data recursively - matches Enhancement handler"""
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

def detect_ring_color(image):
    """Ultra-conservative yellow gold detection - only pure gold colors"""
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    
    # Focus on center 50%
    center_y, center_x = height // 2, width // 2
    crop_size = min(height, width) // 2
    
    y1 = max(0, center_y - crop_size // 2)
    y2 = min(height, center_y + crop_size // 2)
    x1 = max(0, center_x - crop_size // 2)
    x2 = min(width, center_x + crop_size // 2)
    
    center_region = img_array[y1:y2, x1:x2]
    
    # Convert to HSV
    hsv = cv2.cvtColor(center_region, cv2.COLOR_RGB2HSV)
    
    # Calculate average values
    avg_hue = np.mean(hsv[:, :, 0])
    avg_saturation = np.mean(hsv[:, :, 1])
    avg_value = np.mean(hsv[:, :, 2])
    
    # RGB analysis
    r_mean = np.mean(center_region[:, :, 0])
    g_mean = np.mean(center_region[:, :, 1])
    b_mean = np.mean(center_region[:, :, 2])
    
    # Normalize RGB values
    max_rgb = max(r_mean, g_mean, b_mean)
    if max_rgb > 0:
        r_norm = r_mean / max_rgb
        g_norm = g_mean / max_rgb
        b_norm = b_mean / max_rgb
    else:
        r_norm = g_norm = b_norm = 1.0
    
    # Calculate color ratios
    rg_ratio = r_mean / (g_mean + 1)
    rb_ratio = r_mean / (b_mean + 1)
    gb_ratio = g_mean / (b_mean + 1)
    
    # ULTRA-CONSERVATIVE YELLOW GOLD
    is_pure_gold = (
        avg_hue >= 25 and avg_hue <= 32 and
        avg_saturation > 80 and
        avg_value > 120 and avg_value < 200 and
        gb_ratio > 1.4 and
        r_mean > 180 and g_mean > 140 and
        b_mean < 100
    )
    
    if is_pure_gold:
        return "옐로우골드"
    elif rg_ratio > 1.2 and rb_ratio > 1.3 and avg_hue < 15:
        return "로즈골드"
    elif avg_saturation < 50 and avg_value > 180 and b_norm > r_norm:
        return "화이트골드"
    else:
        return "무도금화이트"

def apply_basic_enhancement(image):
    """Apply basic enhancement matching Enhancement V80"""
    if image.mode != 'RGB':
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        else:
            image = image.convert('RGB')
    
    # Match Enhancement V80 basic settings - IMPORTANT!
    brightness = ImageEnhance.Brightness(image)
    image = brightness.enhance(1.12)
    
    contrast = ImageEnhance.Contrast(image)
    image = contrast.enhance(1.08)
    
    color = ImageEnhance.Color(image)
    image = color.enhance(1.05)
    
    return image

def create_thumbnail_with_crop(image, target_size=(1000, 1300)):
    """Create thumbnail with reduced 5% crop for smaller ring"""
    original_width, original_height = image.size
    
    # 5% crop from each side (reduced from 10%)
    crop_percentage = 0.05  # Changed from 0.1 to 0.05
    crop_width = int(original_width * crop_percentage)
    crop_height = int(original_height * crop_percentage)
    
    # Calculate crop box
    left = crop_width
    top = crop_height
    right = original_width - crop_width
    bottom = original_height - crop_height
    
    # Crop image
    cropped = image.crop((left, top, right, bottom))
    
    # Resize to target with high quality
    thumbnail = cropped.resize(target_size, Image.Resampling.LANCZOS)
    
    return thumbnail

def apply_color_specific_enhancement(image, detected_color):
    """Apply color-specific enhancement - Natural colors"""
    if detected_color == "무도금화이트":
        # Natural white enhancement
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.08)
        
        color = ImageEnhance.Color(image)
        image = color.enhance(0.7)  # Keep 70% of original color
        
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.03)
        
    elif detected_color == "옐로우골드":
        # Yellow gold - warm enhancement
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.06)
        
        color = ImageEnhance.Color(image)
        image = color.enhance(1.12)
        
        # Subtle warmth
        img_array = np.array(image)
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 1.03, 0, 255)
        img_array[:, :, 1] = np.clip(img_array[:, :, 1] * 1.02, 0, 255)
        image = Image.fromarray(img_array.astype(np.uint8))
        
    elif detected_color == "로즈골드":
        # Rose gold - pink enhancement
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.05)
        
        color = ImageEnhance.Color(image)
        image = color.enhance(1.05)
        
        # Very subtle pink tone
        img_array = np.array(image)
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 1.02, 0, 255)
        image = Image.fromarray(img_array.astype(np.uint8))
        
    elif detected_color == "화이트골드":
        # White gold - cool metallic
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.06)
        
        color = ImageEnhance.Color(image)
        image = color.enhance(0.85)
    
    return image

def apply_center_vignette(image):
    """Apply VERY subtle center vignette"""
    width, height = image.size
    
    # Create radial gradient
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)
    
    # Distance from center
    distance = np.sqrt(X**2 + Y**2)
    
    # Ultra subtle vignette
    vignette = 1 - (0.015 * np.clip(distance - 0.5, 0, 1))
    vignette = np.clip(vignette, 0.985, 1.0)
    
    # Apply vignette
    img_array = np.array(image)
    for i in range(3):
        img_array[:, :, i] = img_array[:, :, i] * vignette
    
    return Image.fromarray(img_array.astype(np.uint8))

def image_to_base64(image):
    """Convert image to base64 without padding for Make.com"""
    buffered = BytesIO()
    
    if image.mode == 'RGBA':
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
    
    image.save(buffered, format='PNG', quality=95)
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    # Remove padding for Make.com
    img_base64_no_padding = img_base64.rstrip('=')
    
    return img_base64_no_padding

def handler(event):
    """Thumbnail handler function"""
    try:
        logger.info(f"Thumbnail {VERSION} started")
        
        # Find input data
        input_data = find_input_data(event)
        
        if not input_data:
            raise ValueError("No input data found")
        
        # Get image
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
            raise ValueError("Failed to load image")
        
        logger.info(f"Image loaded: {image.size}")
        
        # 1. Apply basic enhancement (matching Enhancement handler) - IMPORTANT!
        enhanced_image = apply_basic_enhancement(image)
        
        # 2. Create thumbnail with 5% crop for smaller ring (90% size)
        thumbnail = create_thumbnail_with_crop(enhanced_image)
        
        # 3. Detect color with ultra-conservative logic
        detected_color = detect_ring_color(thumbnail)
        logger.info(f"Detected color: {detected_color}")
        
        # 4. Apply color-specific enhancement (Natural colors)
        thumbnail = apply_color_specific_enhancement(thumbnail, detected_color)
        
        # 5. Apply very subtle vignette
        thumbnail = apply_center_vignette(thumbnail)
        
        # 6. Very minimal final brightness
        brightness = ImageEnhance.Brightness(thumbnail)
        thumbnail = brightness.enhance(1.01)
        
        # 7. Light sharpness for details
        sharpness = ImageEnhance.Sharpness(thumbnail)
        thumbnail = sharpness.enhance(1.15)
        
        # Convert to base64
        thumbnail_base64 = image_to_base64(thumbnail)
        
        # Return result
        return {
            "output": {
                "thumbnail": thumbnail_base64,
                "size": list(thumbnail.size),
                "detected_color": detected_color,
                "format": "base64_no_padding",
                "version": VERSION,
                "status": "success"
            }
        }
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
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
