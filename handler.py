import runpod
import base64
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2
import requests
import traceback

VERSION = "v53"
THUMBNAIL_WIDTH = 1000
THUMBNAIL_HEIGHT = 1300

def find_input_url(event):
    """Find input URL from various possible locations"""
    # Priority order - check enhanced_image first!
    url_keys = [
        'enhanced_image', 'input_url', 'image_url', 'url', 
        'image', 'enhancedImage', 'imageUrl'
    ]
    
    # Check direct keys
    for key in url_keys:
        if key in event and isinstance(event[key], str):
            url = event[key]
            if url.startswith(('http', 'data:')):
                print(f"Found URL at: {key}")
                return url
    
    # Check input dict
    if 'input' in event and isinstance(event['input'], dict):
        for key in url_keys:
            if key in event['input'] and isinstance(event['input'][key], str):
                url = event['input'][key]
                if url.startswith(('http', 'data:')):
                    print(f"Found URL at: input.{key}")
                    return url
    
    # Check numbered output structure
    if 'output' in event and isinstance(event['output'], dict):
        if 'output' in event['output'] and isinstance(event['output']['output'], dict):
            for key in url_keys:
                if key in event['output']['output']:
                    url = event['output']['output'][key]
                    if isinstance(url, str) and url.startswith(('http', 'data:')):
                        print(f"Found URL at: output.output.{key}")
                        return url
    
    print("No URL found - dumping structure")
    print(f"Event keys: {list(event.keys()) if isinstance(event, dict) else 'Not dict'}")
    if 'input' in event:
        print(f"Input keys: {list(event['input'].keys()) if isinstance(event['input'], dict) else 'Not dict'}")
    
    return None

def load_image_from_url(url):
    """Load image from URL or data URL"""
    try:
        if url.startswith('data:'):
            # Handle data URL
            header, encoded = url.split(',', 1)
            data = base64.b64decode(encoded + '==')  # Add padding just in case
            image = Image.open(BytesIO(data))
        else:
            # Handle HTTP URL
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
        
        # Convert to RGB if needed
        if image.mode not in ('RGB', 'RGBA'):
            image = image.convert('RGB')
        
        return image
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        raise

def detect_wedding_rings(image):
    """Detect wedding rings and find center point"""
    # Convert PIL to OpenCV
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Create detection image (max 1500px for speed)
    height, width = img_cv.shape[:2]
    scale = 1.0
    if max(height, width) > 1500:
        scale = 1500 / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        detect_img = cv2.resize(img_cv, (new_width, new_height))
    else:
        detect_img = img_cv
    
    # Convert to grayscale
    gray = cv2.cvtColor(detect_img, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while keeping edges
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Detect circles (rings)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=int(20 * scale),
        maxRadius=int(200 * scale)
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Scale back to original size
        circles[0, :, :2] = circles[0, :, :2] / scale
        
        # Find center of rings
        if len(circles[0]) >= 2:
            # Use center of two largest circles
            sorted_circles = sorted(circles[0], key=lambda x: x[2], reverse=True)
            center_x = int((sorted_circles[0][0] + sorted_circles[1][0]) / 2)
            center_y = int((sorted_circles[0][1] + sorted_circles[1][1]) / 2)
        else:
            # Use center of largest circle
            center_x = int(circles[0][0][0])
            center_y = int(circles[0][0][1])
        
        print(f"Rings detected at ({center_x}, {center_y})")
        return center_x, center_y, True
    
    # Fallback to image center
    print("No rings detected, using center")
    return width // 2, height // 2, False

def detect_metal_color_conservative(image, ring_area=None):
    """Detect metal color with white/white_gold priority"""
    img_array = np.array(image)
    
    # Focus on ring area if provided
    if ring_area:
        x, y, w, h = ring_area
        img_array = img_array[y:y+h, x:x+w]
    
    # Convert to HSV
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    # Also check RGB values for white detection
    avg_color = np.mean(img_array.reshape(-1, 3), axis=0)
    brightness = np.mean(avg_color)
    
    # 1. WHITE/WHITE_GOLD DETECTION FIRST (Priority)
    # Very bright and low saturation = white gold
    white_lower = np.array([0, 0, 200])      # Any hue, low sat, high value
    white_upper = np.array([180, 30, 255])   # Saturation max 30!
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    white_pixels = cv2.countNonZero(white_mask)
    
    # If very bright and low saturation, it's white/white_gold
    if brightness > 200 and white_pixels > (img_array.shape[0] * img_array.shape[1] * 0.3):
        return 'white'  # 무도금화이트
    
    # 2. ROSE GOLD DETECTION
    # Pink/rose tones
    rose_lower1 = np.array([0, 50, 100])    # Red side
    rose_upper1 = np.array([10, 150, 255])
    rose_lower2 = np.array([170, 50, 100])  # Red wraparound
    rose_upper2 = np.array([180, 150, 255])
    
    rose_mask1 = cv2.inRange(hsv, rose_lower1, rose_upper1)
    rose_mask2 = cv2.inRange(hsv, rose_lower2, rose_upper2)
    rose_mask = cv2.bitwise_or(rose_mask1, rose_mask2)
    rose_pixels = cv2.countNonZero(rose_mask)
    
    # 3. YELLOW GOLD DETECTION (Conservative)
    # Only strong yellow/gold colors
    yellow_lower = np.array([20, 100, 100])   # Saturation min 100!
    yellow_upper = np.array([30, 255, 200])   # Not too bright
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    yellow_pixels = cv2.countNonZero(yellow_mask)
    
    # Decision logic
    total_pixels = img_array.shape[0] * img_array.shape[1]
    
    # Need significant yellow with high saturation for yellow gold
    if yellow_pixels > (total_pixels * 0.2):
        # Double check saturation
        yellow_region = hsv[yellow_mask > 0]
        if len(yellow_region) > 0:
            avg_saturation = np.mean(yellow_region[:, 1])
            if avg_saturation > 100:  # High saturation = real gold color
                return 'yellow_gold'
    
    # Rose gold check
    if rose_pixels > (total_pixels * 0.15):
        return 'rose_gold'
    
    # Default to white for any ambiguous cases
    return 'white'

def create_professional_thumbnail(image, center_x, center_y, metal_color):
    """Create professional quality thumbnail with proper crop and enhancement"""
    width, height = image.size
    
    # Calculate crop box (1000x1300 centered on rings)
    left = max(0, center_x - THUMBNAIL_WIDTH // 2)
    top = max(0, center_y - THUMBNAIL_HEIGHT // 2)
    right = min(width, left + THUMBNAIL_WIDTH)
    bottom = min(height, top + THUMBNAIL_HEIGHT)
    
    # Adjust if hitting boundaries
    if right - left < THUMBNAIL_WIDTH:
        if left == 0:
            right = min(width, THUMBNAIL_WIDTH)
        else:
            left = max(0, right - THUMBNAIL_WIDTH)
    
    if bottom - top < THUMBNAIL_HEIGHT:
        if top == 0:
            bottom = min(height, THUMBNAIL_HEIGHT)
        else:
            top = max(0, bottom - THUMBNAIL_HEIGHT)
    
    # Crop
    cropped = image.crop((left, top, right, bottom))
    
    # Resize to exact dimensions if needed
    if cropped.size != (THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT):
        cropped = cropped.resize((THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT), Image.Resampling.LANCZOS)
    
    # Apply color-specific enhancement
    if metal_color == 'white':
        # White/White Gold - bright and cool
        enhancer = ImageEnhance.Brightness(cropped)
        enhanced = enhancer.enhance(1.25)  # Very bright
        
        enhancer = ImageEnhance.Color(enhanced)
        enhanced = enhancer.enhance(0.9)  # Slightly desaturated
        
        # Cool tone adjustment using numpy
        img_array = np.array(enhanced)
        img_array[:,:,0] = np.clip(img_array[:,:,0] * 0.98, 0, 255)  # Reduce red
        img_array[:,:,2] = np.clip(img_array[:,:,2] * 1.02, 0, 255)  # Increase blue
        enhanced = Image.fromarray(img_array.astype('uint8'))
        
    elif metal_color == 'rose_gold':
        # Rose Gold - warm pink tones
        enhancer = ImageEnhance.Brightness(cropped)
        enhanced = enhancer.enhance(1.15)
        
        enhancer = ImageEnhance.Color(enhanced)
        enhanced = enhancer.enhance(1.1)  # Enhance pink tones
        
    else:  # yellow_gold
        # Yellow Gold - warm and rich
        enhancer = ImageEnhance.Brightness(cropped)
        enhanced = enhancer.enhance(1.1)
        
        enhancer = ImageEnhance.Color(enhanced)
        enhanced = enhancer.enhance(1.05)
    
    # Common enhancements for all
    # Contrast
    enhancer = ImageEnhance.Contrast(enhanced)
    enhanced = enhancer.enhance(1.15)
    
    # Sharpness for detail
    enhancer = ImageEnhance.Sharpness(enhanced)
    enhanced = enhancer.enhance(1.8)  # Strong sharpening
    
    # Edge enhancement
    enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    
    return enhanced

def handler(job):
    """RunPod handler function"""
    print(f"=== Thumbnail Handler {VERSION} Started ===")
    
    try:
        # Find input URL
        input_url = find_input_url(job)
        if not input_url:
            print(f"Job structure: {job}")
            return {
                "output": {
                    "error": "No input URL found",
                    "success": False,
                    "version": VERSION
                }
            }
        
        print(f"Found URL: {input_url[:100]}...")
        
        # Load image
        try:
            image = load_image_from_url(input_url)
            print(f"Image loaded: {image.size}")
        except Exception as e:
            return {
                "output": {
                    "error": f"Failed to load image: {str(e)}",
                    "success": False,
                    "version": VERSION
                }
            }
        
        # Detect rings
        center_x, center_y, rings_found = detect_wedding_rings(image)
        
        # Detect metal color (conservative, white-first)
        ring_area = (
            max(0, center_x - 200),
            max(0, center_y - 200),
            400, 400
        )
        metal_color = detect_metal_color_conservative(image, ring_area)
        print(f"Detected metal color: {metal_color}")
        
        # Create professional thumbnail
        thumbnail = create_professional_thumbnail(image, center_x, center_y, metal_color)
        
        # Convert to base64
        buffer = BytesIO()
        thumbnail.save(buffer, format='JPEG', quality=95)
        buffer.seek(0)
        
        # Base64 encode with padding (for Google Script)
        thumb_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return {
            "output": {
                "success": True,
                "thumbnail": thumb_base64,
                "detected_color": metal_color,
                "rings_found": rings_found,
                "crop_center": f"({center_x}, {center_y})",
                "version": VERSION,
                "message": "Professional thumbnail created successfully"
            }
        }
        
    except Exception as e:
        print(f"Handler error: {str(e)}")
        print(traceback.format_exc())
        return {
            "output": {
                "error": str(e),
                "success": False,
                "version": VERSION,
                "traceback": traceback.format_exc()
            }
        }

runpod.serverless.start({"handler": handler})
