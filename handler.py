import runpod
import base64
import requests
import time
import json
from PIL import Image, ImageEnhance, ImageFilter
import io
import numpy as np
import cv2
from typing import Optional, Dict, Any, Union, Tuple

def find_input_data(data: Dict[str, Any]) -> Optional[Union[str, Dict]]:
    """Find input data from various possible locations - Enhanced for thumbnail"""
    
    if isinstance(data, dict):
        # Priority 1: Check for enhanced_image (from Enhancement step)
        enhanced_keys = ['enhanced_image', 'enhancedImage', 'enhanced', 'image_enhanced']
        for key in enhanced_keys:
            if key in data:
                return data[key]
        
        # Priority 2: Check Make.com pattern (numbered keys)
        for i in range(10):
            num_key = str(i)
            if num_key in data and isinstance(data[num_key], dict):
                # Check nested structure like 4.data.output.output.enhanced_image
                if 'data' in data[num_key]:
                    if 'output' in data[num_key]['data']:
                        if 'output' in data[num_key]['data']['output']:
                            output = data[num_key]['data']['output']['output']
                            for key in enhanced_keys:
                                if key in output:
                                    print(f"Found in numbered structure: {num_key}.data.output.output.{key}")
                                    return output[key]
        
        # Priority 3: Standard image keys
        image_keys = ['image_base64', 'imageBase64', 'image', 'base64', 'url', 'image_url', 'imageUrl']
        for key in image_keys:
            if key in data:
                return data[key]
        
        # Priority 4: Check input sub-object
        if 'input' in data:
            result = find_input_data(data['input'])
            if result:
                return result
        
        # Priority 5: Deep search
        for key, value in data.items():
            if isinstance(value, dict) and key not in ['output', 'error']:
                result = find_input_data(value)
                if result:
                    return result
            elif isinstance(value, str) and len(value) > 100:
                # Might be base64 or URL
                if value.startswith('http') or 'base64' in value or not value.startswith('{'):
                    return value
    
    return None

def download_image_from_url(url: str) -> Image.Image:
    """Download image from URL with retries"""
    headers_list = [
        {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'},
        {'User-Agent': 'Python-Requests/2.31.0'},
        {}
    ]
    
    for headers in headers_list:
        try:
            response = requests.get(url, headers=headers, timeout=30, stream=True)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert('RGB')
        except Exception as e:
            print(f"Failed with headers {headers}: {e}")
            continue
            
    raise ValueError(f"Failed to download image from URL: {url}")

def base64_to_image(base64_str: str) -> Image.Image:
    """Convert base64 string to PIL Image with padding fix"""
    # Remove data URL prefix if present
    if 'base64,' in base64_str:
        base64_str = base64_str.split('base64,')[1]
    
    # Fix padding if needed
    padding = 4 - len(base64_str) % 4
    if padding != 4:
        base64_str += '=' * padding
        print(f"Added {padding} padding characters")
    
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data)).convert('RGB')

def apply_basic_enhancement(image: Image.Image) -> Image.Image:
    """Apply V60 basic enhancement matching Enhancement Handler"""
    # Convert to numpy array
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # V60 adjustments
    brightness = 1.45  # 45% brightness increase
    contrast = 1.05
    gamma = 0.6
    saturation = 0.65  # 35% saturation decrease
    additional_brightness = 0.02  # 2% additional brightness
    
    # Apply brightness
    img_array = img_array * brightness
    
    # Apply contrast
    img_array = (img_array - 0.5) * contrast + 0.5
    
    # Apply gamma correction
    img_array = np.power(np.clip(img_array, 0, 1), gamma)
    
    # Apply saturation adjustment
    img_array_uint8 = np.clip(img_array * 255, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_array_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:,:,1] = hsv[:,:,1] * saturation
    hsv = np.clip(hsv, 0, 255)
    img_array_uint8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    img_array = img_array_uint8.astype(np.float32) / 255.0
    
    # Apply additional brightness
    img_array = img_array + additional_brightness
    
    # Ensure values are in valid range
    img_array = np.clip(img_array, 0, 1)
    
    # Convert back to PIL Image
    img_array = (img_array * 255).astype(np.uint8)
    return Image.fromarray(img_array)

def detect_ring_color(image: Image.Image) -> str:
    """Detect ring color from image with improved accuracy"""
    # Convert to numpy array
    img_array = np.array(image)
    
    # Get center region (where ring is likely to be)
    h, w = img_array.shape[:2]
    center_y, center_x = h // 2, w // 2
    region_size = min(h, w) // 3
    
    center_region = img_array[
        center_y - region_size:center_y + region_size,
        center_x - region_size:center_x + region_size
    ]
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(center_region, cv2.COLOR_RGB2HSV)
    
    # Define color ranges (무도금화이트 우선)
    white_mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 30, 255]))
    yellow_mask = cv2.inRange(hsv, np.array([15, 50, 100]), np.array([35, 255, 255]))
    rose_mask = cv2.inRange(hsv, np.array([0, 30, 100]), np.array([15, 255, 255]))
    
    # Count pixels for each color
    white_pixels = cv2.countNonZero(white_mask)
    yellow_pixels = cv2.countNonZero(yellow_mask)
    rose_pixels = cv2.countNonZero(rose_mask)
    
    # Calculate ratios
    total_pixels = center_region.shape[0] * center_region.shape[1]
    white_ratio = white_pixels / total_pixels
    yellow_ratio = yellow_pixels / total_pixels
    rose_ratio = rose_pixels / total_pixels
    
    print(f"Color ratios - White: {white_ratio:.2f}, Yellow: {yellow_ratio:.2f}, Rose: {rose_ratio:.2f}")
    
    # Check for white/silver first (including 무도금화이트)
    if white_ratio > 0.3:
        # Additional brightness check
        gray = cv2.cvtColor(center_region, cv2.COLOR_RGB2GRAY)
        avg_brightness = np.mean(gray)
        if avg_brightness > 180:
            return '무도금화이트'
        else:
            return '화이트골드'
    
    # Check other colors
    if yellow_ratio > rose_ratio and yellow_ratio > 0.1:
        return '옐로우골드'
    elif rose_ratio > yellow_ratio and rose_ratio > 0.1:
        return '로즈골드'
    else:
        # Default based on overall color tone
        avg_b, avg_g, avg_r = np.mean(center_region, axis=(0, 1))
        if avg_r > avg_g > avg_b:
            return '로즈골드'
        elif avg_r > avg_b and avg_g > avg_b:
            return '옐로우골드'
        else:
            return '화이트골드'

def apply_color_specific_enhancement(image: Image.Image, detected_color: str) -> Image.Image:
    """Apply color-specific enhancement to thumbnail"""
    print(f"Applying enhancement for: {detected_color}")
    
    # Base enhancements
    enhancer = ImageEnhance.Brightness(image)
    
    if detected_color == '무도금화이트':
        # Brightest enhancement for white
        image = enhancer.enhance(1.25)
        # Cool tone adjustment
        img_array = np.array(image).astype(np.float32)
        img_array[:,:,2] *= 1.05  # Slight blue enhancement
        img_array[:,:,0] *= 0.98  # Slight red reduction
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        image = Image.fromarray(img_array)
        # Extra contrast for white metals
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.1)
        
    elif detected_color == '로즈골드':
        # Warm pink enhancement
        image = enhancer.enhance(1.15)
        # Enhance red/pink tones
        img_array = np.array(image).astype(np.float32)
        img_array[:,:,0] *= 1.12  # Red enhancement
        img_array[:,:,1] *= 1.02  # Slight green
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        image = Image.fromarray(img_array)
        # Enhance saturation for rose gold
        saturation = ImageEnhance.Color(image)
        image = saturation.enhance(1.1)
        
    elif detected_color == '옐로우골드':
        # Golden enhancement
        image = enhancer.enhance(1.1)
        # Enhance yellow/gold tones
        img_array = np.array(image).astype(np.float32)
        img_array[:,:,0] *= 1.08  # Red
        img_array[:,:,1] *= 1.06  # Green
        img_array[:,:,2] *= 0.96  # Reduce blue
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        image = Image.fromarray(img_array)
        # Warm saturation
        saturation = ImageEnhance.Color(image)
        image = saturation.enhance(1.08)
        
    else:  # 화이트골드
        # Neutral bright enhancement
        image = enhancer.enhance(1.2)
        # Slight cool tone
        img_array = np.array(image).astype(np.float32)
        img_array[:,:,2] *= 1.02  # Slight blue
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        image = Image.fromarray(img_array)
    
    # Final sharpness enhancement for all
    sharpness = ImageEnhance.Sharpness(image)
    image = sharpness.enhance(1.8)
    
    # Edge enhancement
    image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    
    return image

def create_thumbnail_with_crop(image: Image.Image) -> Image.Image:
    """Create 1000x1300 thumbnail with center crop"""
    # Target size
    target_width = 1000
    target_height = 1300
    target_ratio = target_width / target_height
    
    # Current size
    width, height = image.size
    current_ratio = width / height
    
    print(f"Original size: {width}x{height}, ratio: {current_ratio:.2f}")
    print(f"Target size: {target_width}x{target_height}, ratio: {target_ratio:.2f}")
    
    # Calculate crop to match aspect ratio
    if current_ratio > target_ratio:
        # Image is wider than target ratio
        new_width = int(height * target_ratio)
        left = (width - new_width) // 2
        right = left + new_width
        image = image.crop((left, 0, right, height))
        print(f"Cropped horizontally: {left}, 0, {right}, {height}")
    else:
        # Image is taller than target ratio
        new_height = int(width / target_ratio)
        top = (height - new_height) // 2
        bottom = top + new_height
        image = image.crop((0, top, width, bottom))
        print(f"Cropped vertically: 0, {top}, {width}, {bottom}")
    
    # Resize to exact target size with high quality
    image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    print(f"Resized to: {image.size}")
    
    return image

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """Main handler function for RunPod thumbnail"""
    try:
        print(f"Thumbnail V60 received event: {json.dumps(event, indent=2)[:500]}...")
        
        # Find input data (enhanced image)
        input_data = find_input_data(event)
        
        if not input_data:
            # Try direct access
            if 'input' in event:
                input_data = event['input'].get('enhanced_image') or event['input'].get('image')
            
            if not input_data:
                print("Failed to find input data. Event structure:")
                print(json.dumps(event, indent=2)[:1000])
                return {
                    "output": {
                        "error": "No enhanced image found",
                        "status": "failed",
                        "version": "v60"
                    }
                }
        
        print(f"Found input data type: {type(input_data).__name__}")
        
        # Load image based on input type
        image = None
        
        if isinstance(input_data, str):
            if input_data.startswith('http'):
                print(f"Loading from URL: {input_data[:100]}...")
                image = download_image_from_url(input_data)
            else:
                print("Loading from base64 string")
                image = base64_to_image(input_data)
        elif isinstance(input_data, dict):
            # Try to find image in dict
            for key in ['enhanced_image', 'image', 'base64', 'url', 'data']:
                if key in input_data:
                    if isinstance(input_data[key], str):
                        if input_data[key].startswith('http'):
                            image = download_image_from_url(input_data[key])
                        else:
                            image = base64_to_image(input_data[key])
                        break
        
        if image is None:
            return {
                "output": {
                    "error": "Failed to load image from input",
                    "status": "failed",
                    "version": "v60"
                }
            }
        
        print(f"Image loaded successfully: {image.size}")
        
        # Step 1: Apply basic enhancement (matching Enhancement Handler)
        enhanced_image = apply_basic_enhancement(image)
        print("Basic enhancement applied (V60)")
        
        # Step 2: Create 1000x1300 thumbnail
        thumbnail = create_thumbnail_with_crop(enhanced_image)
        print(f"Thumbnail created: {thumbnail.size}")
        
        # Step 3: Detect ring color from thumbnail
        detected_color = detect_ring_color(thumbnail)
        print(f"Detected ring color: {detected_color}")
        
        # Step 4: Apply color-specific enhancement
        thumbnail = apply_color_specific_enhancement(thumbnail, detected_color)
        print("Color-specific enhancement applied")
        
        # Convert to base64
        buffered = io.BytesIO()
        # Convert RGBA to RGB if needed
        if thumbnail.mode == 'RGBA':
            background = Image.new('RGB', thumbnail.size, (255, 255, 255))
            background.paste(thumbnail, mask=thumbnail.split()[3])
            thumbnail = background
            
        thumbnail.save(buffered, format="PNG", quality=95)
        thumbnail_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Remove padding for Make.com
        thumbnail_base64_no_padding = thumbnail_base64.rstrip('=')
        print(f"Base64 length (no padding): {len(thumbnail_base64_no_padding)}")
        
        # Return with proper structure
        return {
            "output": {
                "thumbnail": thumbnail_base64_no_padding,
                "size": list(thumbnail.size),
                "detected_color": detected_color,
                "original_size": list(image.size),
                "version": "v60_complete",
                "status": "success",
                "process": "enhancement_crop_detect_enhance_v60"
            }
        }
        
    except Exception as e:
        print(f"Error in Thumbnail V60: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "output": {
                "error": str(e),
                "status": "failed",
                "version": "v60",
                "traceback": traceback.format_exc()
            }
        }

# RunPod serverless handler
runpod.serverless.start({"handler": handler})
