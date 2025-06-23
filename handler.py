import runpod
import base64
import numpy as np
from PIL import Image, ImageEnhance
import cv2
import io
import os
import traceback
import time

# Version info
VERSION = "v19-thumbnail"

# Import Replicate when available
try:
    import replicate
    REPLICATE_AVAILABLE = True
except ImportError:
    REPLICATE_AVAILABLE = False
    print(f"[{VERSION}] Replicate not available")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

class ThumbnailProcessorV19:
    """v19 Thumbnail Processor - Simplified but Effective Detection"""
    
    def __init__(self):
        print(f"[{VERSION}] Initializing - Simplified Black Frame Detection")
        self.replicate_client = None
    
    def detect_black_frame_simple(self, image_np):
        """Simple but effective black frame detection"""
        h, w = image_np.shape[:2]
        print(f"[{VERSION}] Detecting black frame in {w}x{h} image")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Method 1: Find large dark regions
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Sort by area
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            for contour in contours[:5]:  # Check top 5 largest contours
                x, y, cw, ch = cv2.boundingRect(contour)
                area = cw * ch
                
                # Check if it's a significant box (at least 5% of image)
                if area > (w * h * 0.05):
                    # Additional check: is it roughly square?
                    aspect_ratio = cw / ch
                    if 0.7 < aspect_ratio < 1.3:  # Roughly square
                        print(f"[{VERSION}] Black box found: ({x},{y}) {cw}x{ch}")
                        
                        # Create mask
                        mask = np.zeros((h, w), dtype=np.uint8)
                        cv2.drawContours(mask, [contour], -1, 255, -1)
                        
                        return {
                            'has_frame': True,
                            'detected': True,
                            'mask': mask,
                            'bbox': (x, y, cw, ch),
                            'method': 'contour'
                        }
        
        # Method 2: Check for frame edges
        edge_thickness = self._check_frame_edges(gray)
        if edge_thickness > 20:
            print(f"[{VERSION}] Edge frame detected: {edge_thickness}px")
            mask = self._create_edge_mask(h, w, edge_thickness)
            return {
                'has_frame': True,
                'detected': True,
                'mask': mask,
                'thickness': edge_thickness,
                'method': 'edge'
            }
        
        print(f"[{VERSION}] No black frame detected")
        return {'has_frame': False, 'detected': False, 'mask': None}
    
    def _check_frame_edges(self, gray):
        """Check for black edges"""
        h, w = gray.shape
        threshold = 50
        
        # Check each edge
        edges = []
        
        # Top edge
        for i in range(min(100, h//4)):
            if np.mean(gray[i, :]) < threshold:
                edges.append(i)
            else:
                break
        top = len(edges)
        
        # Similar for other edges...
        # Return average thickness if significant
        if top > 20:
            return top
        
        return 0
    
    def _create_edge_mask(self, h, w, thickness):
        """Create mask for edge frame"""
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[:thickness, :] = 255  # Top
        mask[-thickness:, :] = 255  # Bottom
        mask[:, :thickness] = 255  # Left
        mask[:, -thickness:] = 255  # Right
        return mask
    
    def remove_black_frame_replicate(self, image, detection_result):
        """Remove black frame using Replicate API"""
        if not detection_result['detected'] or not REPLICATE_AVAILABLE:
            return image
        
        try:
            print(f"[{VERSION}] Removing black frame with Replicate")
            
            # Get mask
            mask_np = detection_result['mask']
            
            # Dilate mask for better results
            kernel = np.ones((10, 10), np.uint8)
            mask_np = cv2.dilate(mask_np, kernel, iterations=2)
            
            mask_img = Image.fromarray(mask_np)
            
            # Convert to base64
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            
            mask_buffer = io.BytesIO()
            mask_img.save(mask_buffer, format='PNG')
            mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode('utf-8')
            
            # Initialize Replicate client
            if not self.replicate_client:
                self.replicate_client = replicate.Client(api_token=os.environ.get('REPLICATE_API_TOKEN'))
            
            # Run inpainting
            output = self.replicate_client.run(
                "stability-ai/stable-diffusion-inpainting",
                input={
                    "image": f"data:image/png;base64,{img_base64}",
                    "mask": f"data:image/png;base64,{mask_base64}",
                    "prompt": "clean white professional product photography background, seamless",
                    "negative_prompt": "black, dark, frame, border, box",
                    "num_inference_steps": 30,
                    "guidance_scale": 8.0
                }
            )
            
            if output and len(output) > 0:
                response = requests.get(output[0])
                result = Image.open(io.BytesIO(response.content))
                print(f"[{VERSION}] Black frame removed successfully")
                return result
            
        except Exception as e:
            print(f"[{VERSION}] Replicate failed: {e}")
            traceback.print_exc()
        
        # Fallback: crop the black area
        if 'bbox' in detection_result:
            x, y, w, h = detection_result['bbox']
            # Crop inside the black box
            return image.crop((x+5, y+5, x+w-5, y+h-5))
        
        return image
    
    def apply_simple_enhancement(self, image):
        """Simple color enhancement - same as enhancement handler"""
        # 1. Brightness
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.1)
        
        # 2. Contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.05)
        
        # 3. Saturation
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.02)
        
        # 4. Background blending
        img_np = np.array(image)
        h, w = img_np.shape[:2]
        
        background_color = (245, 243, 240)
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.rectangle(mask, (30, 30), (w-30, h-30), 1.0, -1)
        mask = cv2.GaussianBlur(mask, (61, 61), 30)
        
        for i in range(3):
            img_np[:, :, i] = img_np[:, :, i] * mask + background_color[i] * (1 - mask) * 0.3
        
        return Image.fromarray(img_np.astype(np.uint8))
    
    def create_thumbnail_with_detail(self, image, target_size=(1000, 1300)):
        """Create thumbnail with proper cropping and detail enhancement"""
        img_np = np.array(image)
        h, w = img_np.shape[:2]
        
        # Find ring bounds using edge detection
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get bounding box of all contours
            all_points = np.concatenate(contours)
            x, y, w_box, h_box = cv2.boundingRect(all_points)
            
            # Add 10% padding
            padding_x = int(w_box * 0.1)
            padding_y = int(h_box * 0.1)
            
            x = max(0, x - padding_x)
            y = max(0, y - padding_y)
            w_box = min(w - x, w_box + 2 * padding_x)
            h_box = min(h - y, h_box + 2 * padding_y)
            
            # Crop to ring area
            cropped = img_np[y:y+h_box, x:x+w_box]
        else:
            # Fallback: center crop
            cropped = img_np
        
        # Resize to target maintaining aspect ratio
        target_w, target_h = target_size
        h_crop, w_crop = cropped.shape[:2]
        
        # Calculate scale to fit target
        scale = min(target_w / w_crop, target_h / h_crop)
        new_w = int(w_crop * scale)
        new_h = int(h_crop * scale)
        
        # Resize
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Create canvas and center the image
        canvas = np.full((target_h, target_w, 3), (245, 243, 240), dtype=np.uint8)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # Convert to PIL and enhance details
        thumb_img = Image.fromarray(canvas)
        
        # Sharpness enhancement
        enhancer = ImageEnhance.Sharpness(thumb_img)
        thumb_img = enhancer.enhance(1.3)
        
        print(f"[{VERSION}] Created {target_w}x{target_h} thumbnail")
        
        return thumb_img

# Global instance
processor_instance = None

def get_processor():
    global processor_instance
    if processor_instance is None:
        processor_instance = ThumbnailProcessorV19()
    return processor_instance

def find_base64_in_dict(data, depth=0, max_depth=10):
    """Find base64 image in nested dictionary"""
    if depth > max_depth:
        return None
    
    if isinstance(data, str) and len(data) > 100:
        return data
    
    if isinstance(data, dict):
        for key in ['image', 'base64', 'data', 'input', 'file']:
            if key in data and isinstance(data[key], str) and len(data[key]) > 100:
                return data[key]
        
        for value in data.values():
            result = find_base64_in_dict(value, depth + 1, max_depth)
            if result:
                return result
    
    elif isinstance(data, list):
        for item in data:
            result = find_base64_in_dict(item, depth + 1, max_depth)
            if result:
                return result
    
    return None

def decode_base64_image(base64_str):
    """Decode base64 string to PIL Image"""
    try:
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        
        base64_str = base64_str.strip()
        
        padding = 4 - len(base64_str) % 4
        if padding != 4:
            base64_str += '=' * padding
        
        img_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_data))
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        return img
        
    except Exception as e:
        print(f"[{VERSION}] Error decoding base64: {e}")
        raise

def encode_image_to_base64(image, format='PNG'):
    """Encode image to base64"""
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        buffer = io.BytesIO()
        image.save(buffer, format=format, quality=95 if format == 'JPEG' else None)
        buffer.seek(0)
        
        # Base64 encoding - remove padding for Make.com
        base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        base64_str = base64_str.rstrip('=')
        
        return base64_str
        
    except Exception as e:
        print(f"[{VERSION}] Error encoding image: {e}")
        raise

def handler(job):
    """RunPod thumbnail handler"""
    try:
        start_time = time.time()
        job_input = job["input"]
        
        print(f"[{VERSION}] Thumbnail processing started")
        print(f"[{VERSION}] REPLICATE_AVAILABLE: {REPLICATE_AVAILABLE}")
        
        # Find base64 image in nested structure
        base64_image = find_base64_in_dict(job_input)
        if not base64_image:
            return {
                "output": {
                    "error": "No image data found",
                    "version": VERSION,
                    "success": False
                }
            }
        
        # Decode image
        image = decode_base64_image(base64_image)
        print(f"[{VERSION}] Image decoded: {image.size}")
        
        # numpy conversion
        image_np = np.array(image)
        
        # 1. Detect black frame (simplified)
        processor = get_processor()
        frame_info = processor.detect_black_frame_simple(image_np)
        
        # 2. Remove frame if detected
        if frame_info['has_frame']:
            print(f"[{VERSION}] Black frame detected - removing")
            image = processor.remove_black_frame_replicate(image, frame_info)
        
        # 3. Apply color enhancement
        image = processor.apply_simple_enhancement(image)
        
        # 4. Create thumbnail with detail enhancement
        thumbnail = processor.create_thumbnail_with_detail(image, (1000, 1300))
        
        # Encode result
        thumbnail_base64 = encode_image_to_base64(thumbnail)
        
        # Processing time
        processing_time = time.time() - start_time
        print(f"[{VERSION}] Processing completed in {processing_time:.2f}s")
        
        # Return structure
        return {
            "output": {
                "thumbnail": thumbnail_base64,
                "has_black_frame": frame_info['has_frame'],
                "success": True,
                "version": VERSION,
                "processing_time": round(processing_time, 2),
                "original_size": list(image.size),
                "thumbnail_size": [1000, 1300]
            }
        }
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"[{VERSION}] {error_msg}")
        traceback.print_exc()
        
        return {
            "output": {
                "error": error_msg,
                "success": False,
                "version": VERSION
            }
        }

# RunPod start
if __name__ == "__main__":
    print("="*70)
    print(f"Wedding Ring Thumbnail {VERSION}")
    print("Thumbnail Handler (b_file)")
    print(f"Replicate Available: {REPLICATE_AVAILABLE}")
    print("="*70)
    
    runpod.serverless.start({"handler": handler})
