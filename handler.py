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

VERSION = "V72-ImprovedColorDetection"

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

def detect_ring_color(image):
    """Improved color detection matching Enhancement V72"""
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
    
    # Convert to HSV for better color analysis
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
    rg_ratio = r_mean / (g_mean + 1)  # Red to Green ratio
    rb_ratio = r_mean / (b_mean + 1)  # Red to Blue ratio
    gb_ratio = g_mean / (b_mean + 1)  # Green to Blue ratio
    
    # Improved color detection logic
    if avg_saturation < 15 and avg_value > 220:
        # Very low saturation + high brightness = 무도금화이트
        return "무도금화이트"
    elif avg_saturation < 30:
        # Low saturation = 화이트골드
        return "화이트골드"
    elif avg_hue >= 15 and avg_hue <= 35 and gb_ratio > 1.1:
        # Hue in yellow range AND green > blue = 옐로우골드
        return "옐로우골드"
    elif rg_ratio > 1.15 and avg_hue < 20:
        # High red ratio with low hue = 로즈골드
        return "로즈골드"
    elif gb_ratio > 1.05:
        # Green slightly higher than blue = 옐로우골드
        return "옐로우골드"
    else:
        # Default based on warmth
        if r_norm > 0.95 and g_norm > 0.85:
            return "옐로우골드"
        elif r_norm > g_norm and r_norm > b_norm:
            return "로즈골드"
        else:
            return "화이트골드"

def apply_basic_enhancement(image):
    """Apply basic enhancement matching Enhancement V72"""
    if image.mode != 'RGB':
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        else:
            image = image.convert('RGB')
    
    # Match Enhancement V72 basic settings
    brightness = ImageEnhance.Brightness(image)
    image = brightness.enhance(1.12)
    
    contrast = ImageEnhance.Contrast(image)
    image = contrast.enhance(1.08)
    
    color = ImageEnhance.Color(image)
    image = color.enhance(1.05)
    
    return image

def create_thumbnail_with_crop(image, target_size=(1000, 1300)):
    """Create thumbnail with minimal 10% crop"""
    original_width, original_height = image.size
    
    # 10% crop from each side
    crop_percentage = 0.1
    crop_width = int(original_width * crop_percentage)
    crop_height = int(original_height * crop_percentage)
    
    # Calculate crop box
    left = crop_width
    top = crop_height
    right = original_width - crop_width
    bottom = original_height - crop_height
    
    # Crop image
    cropped = image.crop((left, top, right, bottom))
    
    # Resize to target
    thumbnail = cropped.resize(target_size, Image.Resampling.LANCZOS)
    
    return thumbnail

def apply_color_specific_enhancement(image, detected_color):
    """Apply light color-specific enhancement for thumbnail"""
    if detected_color == "무도금화이트":
        # Strong pure white enhancement
        img_array = np.array(image)
        
        # Convert to LAB for better white control
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # Increase lightness significantly
        lab[:, :, 0] = np.clip(lab[:, :, 0] * 1.2, 0, 255)
        
        # Remove all color (pure white)
        lab[:, :, 1] = lab[:, :, 1] * 0.3  # Nearly eliminate a channel
        lab[:, :, 2] = lab[:, :, 2] * 0.3  # Nearly eliminate b channel
        
        # Convert back
        lab = lab.astype(np.uint8)
        img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        image = Image.fromarray(img_array)
        
        # Additional brightness
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.1)
        
        # Remove saturation completely
        color = ImageEnhance.Color(image)
        image = color.enhance(0.2)
        
        # Blue boost to eliminate yellow
        img_array = np.array(image)
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 1.1, 0, 255)
        image = Image.fromarray(img_array)
        
    elif detected_color == "옐로우골드":
        # Light warm enhancement
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.05)
        
    elif detected_color == "로즈골드":
        # Light pink enhancement
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.03)
        
    elif detected_color == "화이트골드":
        # Light cool enhancement
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(1.06)
    
    return image

def apply_center_vignette(image):
    """Apply very subtle center vignette"""
    width, height = image.size
    
    # Create radial gradient
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)
    
    # Distance from center
    distance = np.sqrt(X**2 + Y**2)
    
    # Very subtle vignette (0.05 strength)
    vignette = 1 - (0.05 * np.clip(distance - 0.5, 0, 1))
    vignette = np.clip(vignette, 0.95, 1.0)  # Minimum brightness 95%
    
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
        
        # 1. Apply basic enhancement (matching Enhancement handler)
        enhanced_image = apply_basic_enhancement(image)
        
        # 2. Create thumbnail with minimal crop (10%)
        thumbnail = create_thumbnail_with_crop(enhanced_image)
        
        # 3. Detect color with improved logic
        detected_color = detect_ring_color(thumbnail)
        logger.info(f"Detected color: {detected_color}")
        
        # 4. Apply color-specific enhancement
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
