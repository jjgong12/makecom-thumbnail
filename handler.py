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
    """Apply V61 basic enhancement matching Enhancement Handler - Natural tone"""
    # Convert to numpy array
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # V61 adjustments - natural tone
    brightness = 1.25  # Reduced from 1.45
    contrast = 1.05
    gamma = 0.8  # Increased from 0.6 for natural look
    saturation = 0.8  # 20% saturation decrease instead of 35%
    
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
    
    # Apply subtle LAB brightening
    lab = cv2.cvtColor((img_array * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[:,:,0] = lab[:,:,0] * 1.05  # Subtle brightening
    lab = np.clip(lab, 0, 255)
    img_array = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    
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
    """Apply color-specific enhancement to thumbnail - V61 natural tone"""
    print(f"Applying enhancement for: {detected_color}")
    
    # Base enhancements - more subtle for natural look
    enhancer = ImageEnhance.Brightness(image)
    
    if detected_color == '무도금화이트':
        # Moderate enhancement for white
        image = enhancer.enhance(1.15)  # Reduced from 1.25
        # Cool tone adjustment
        img_array = np.array(image).astype(np.float32)
        img_array[:,:,2] *= 1.03  # Subtle blue enhancement
        img_array[:,:,0] *= 0.99  # Very slight red reduction
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        image = Image.fromarray(img_array)
        # Moderate contrast
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(1.05)  # Reduced from 1.1
        
    elif detected_color == '로즈골드':
        # Warm pink enhancement
        image = enhancer.enhance(1.1)  # Reduced from 1.15
        # Enhance red/pink tones
        img_array = np.array(image).astype(np.float32)
        img_array[:,:,0] *= 1.08  # Red enhancement (reduced from 1.12)
        img_array[:,:,1] *= 1.01  # Very slight green
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        image = Image.fromarray(img_array)
        # Moderate saturation
        saturation = ImageEnhance.Color(image)
        image = saturation.enhance(1.05)  # Reduced from 1.1
        
    elif detected_color == '옐로우골드':
        # Golden enhancement
        image = enhancer.enhance(1.08)  # Reduced from 1.1
        # Enhance yellow/gold tones
        img_array = np.array(image).astype(np.float32)
        img_array[:,:,0] *= 1.05  # Red (reduced)
        img_array[:,:,1] *= 1.04  # Green (reduced)
        img_array[:,:,2] *= 0.98  # Slight blue reduction
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        image = Image.fromarray(img_array)
        # Subtle saturation
        saturation = ImageEnhance.Color(image)
        image = saturation.enhance(1.04)  # Reduced from 1.08
        
    else:  # 화이트골드
        # Neutral bright enhancement
        image = enhancer.enhance(1.12)  # Reduced from 1.2
        # Very slight cool tone
        img_array = np.array(image).astype(np.float32)
        img_array[:,:,2] *= 1.01  # Very subtle blue
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        image = Image.fromarray(img_array)
    
    # Final sharpness enhancement - moderate for natural look
    sharpness = ImageEnhance.Sharpness(image)
    image = sharpness.enhance(1.5)  # Reduced from 1.8
    
    # Gentle edge enhancement
    image = image.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=3))
    
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

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """Main handler function for RunPod thumbnail - Fixed parameter name"""
    try:
        # Get job input
        job_input = job.get("input", {})
        print(f"Thumbnail V61 received input: {json.dumps(job_input, indent=2)[:500]}...")
        
        # Find input data (enhanced image)
        input_data = find_input_data(job_input)
        
        # If not found in job_input, try the whole job dict
        if not input_data:
            input_data = find_input_data(job)
        
        if not input_data:
            # Last resort - check if job_input itself is the image data
            if isinstance(job_input, str) and len(job_input) > 100:
                input_data = job_input
            else:
                print("Failed to find input data. Job structure:")
                print(json.dumps(job, indent=2)[:1000])
                return {
                    "output": {
                        "error": "No enhanced image found",
                        "status": "failed",
                        "version": "v61"
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
            for key in ['enhanced_image', 'enhancedImage', 'image', 'base64', 'url', 'data']:
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
                    "version": "v61"
                }
            }
        
        print(f"Image loaded successfully: {image.size}")
        
        # Step 1: Apply basic enhancement (matching Enhancement Handler)
        enhanced_image = apply_basic_enhancement(image)
        print("Basic enhancement applied (V61 natural tone)")
        
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
                "version": "v61_natural",
                "status": "success",
                "process": "enhancement_crop_detect_enhance_v61"
            }
        }
        
    except Exception as e:
        print(f"Error in Thumbnail V61: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "output": {
                "error": str(e),
                "status": "failed",
                "version": "v61",
                "traceback": traceback.format_exc()
            }
        }

# RunPod serverless handler
runpod.serverless.start({"handler": handler})
