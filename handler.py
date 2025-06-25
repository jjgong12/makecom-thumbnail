import json
import base64
import traceback
import replicate
import requests
from io import BytesIO
from typing import Dict, Any, Optional
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import time

VERSION = "45"

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler for wedding ring thumbnail creation
    v45: Simplified to avoid timeouts
    """
    start_time = time.time()
    print(f"Thumbnail Handler v{VERSION} starting at {start_time}")
    
    try:
        # Simple direct check first
        input_url = None
        
        # Check most common locations
        if 'input' in event and isinstance(event['input'], dict):
            for key in ['enhanced_image', 'image_url', 'url', 'image']:
                if key in event['input'] and isinstance(event['input'][key], str):
                    val = event['input'][key]
                    if val.startswith(('http', 'data:')):
                        input_url = val
                        print(f"Found URL in input.{key}")
                        break
        
        # Direct check
        if not input_url:
            for key in ['enhanced_image', 'image_url', 'url', 'image']:
                if key in event and isinstance(event[key], str):
                    val = event[key]
                    if val.startswith(('http', 'data:')):
                        input_url = val
                        print(f"Found URL in {key}")
                        break
        
        # Check numbered paths (simplified)
        if not input_url:
            for i in range(5):  # Limit to 5
                try:
                    # Try common path pattern
                    if str(i) in event:
                        obj = event[str(i)]
                        if isinstance(obj, dict) and 'data' in obj:
                            obj = obj['data']
                            if isinstance(obj, dict) and 'output' in obj:
                                obj = obj['output']
                                if isinstance(obj, dict) and 'output' in obj:
                                    obj = obj['output']
                                    if isinstance(obj, dict) and 'enhanced_image' in obj:
                                        val = obj['enhanced_image']
                                        if isinstance(val, str) and val.startswith(('http', 'data:')):
                                            input_url = val
                                            print(f"Found URL at {i}.data.output.output.enhanced_image")
                                            break
                except:
                    continue
        
        if not input_url:
            print("ERROR: No input URL found")
            print(f"Event keys: {list(event.keys())}")
            return {
                "output": {
                    "status": "error",
                    "error": "No input URL found",
                    "version": VERSION
                }
            }
        
        print(f"Processing image from URL type: {'data URL' if input_url.startswith('data:') else 'HTTP URL'}")
        
        # Process the image
        result = process_image(input_url)
        
        end_time = time.time()
        print(f"Total processing time: {end_time - start_time:.2f} seconds")
        
        return result
        
    except Exception as e:
        print(f"ERROR in handler: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return {
            "output": {
                "status": "error",
                "error": f"Handler error: {str(e)}",
                "version": VERSION
            }
        }


def load_image_from_url(url: str) -> Image.Image:
    """Load image from URL or data URL"""
    try:
        if url.startswith('data:'):
            # Handle data URL
            header, data = url.split(',', 1)
            data = ''.join(data.split())
            # Add padding if needed
            padding = 4 - len(data) % 4
            if padding != 4:
                data += '=' * padding
            img_data = base64.b64decode(data)
            return Image.open(BytesIO(img_data))
        else:
            # Handle HTTP URL
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"ERROR loading image: {str(e)}")
        raise


def simple_black_frame_detection(img: Image.Image) -> Dict[str, Any]:
    """Simplified but effective black frame detection"""
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    
    # Check edges with low threshold
    edge_size = min(200, min(h, w) // 10)
    threshold = 30
    
    # Check each edge
    edges = {
        'top': np.mean(gray[:edge_size, :]) < threshold,
        'bottom': np.mean(gray[-edge_size:, :]) < threshold,
        'left': np.mean(gray[:, :edge_size]) < threshold,
        'right': np.mean(gray[:, -edge_size:]) < threshold
    }
    
    # If 3 or more edges are dark, it's likely a black frame
    dark_edges = sum(edges.values())
    
    if dark_edges >= 3:
        # Calculate approximate thickness
        thickness = 0
        
        # Top
        for i in range(min(300, h//3)):
            if np.mean(gray[i, :]) > threshold:
                thickness = max(thickness, i)
                break
        
        # Bottom
        for i in range(min(300, h//3)):
            if np.mean(gray[h-i-1, :]) > threshold:
                thickness = max(thickness, i)
                break
        
        # Left
        for i in range(min(300, w//3)):
            if np.mean(gray[:, i]) > threshold:
                thickness = max(thickness, i)
                break
        
        # Right
        for i in range(min(300, w//3)):
            if np.mean(gray[:, w-i-1]) > threshold:
                thickness = max(thickness, i)
                break
        
        if thickness == 0:
            thickness = 100  # Default
        
        return {
            'detected': True,
            'thickness': thickness,
            'dark_edges': dark_edges
        }
    
    return {
        'detected': False,
        'thickness': 0,
        'dark_edges': dark_edges
    }


def remove_black_frame_simple(img: Image.Image, thickness: int) -> Image.Image:
    """Simple but effective black frame removal"""
    try:
        print(f"Attempting Replicate inpainting with thickness: {thickness}")
        
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        
        # Create mask
        mask = np.zeros((h, w), dtype=np.uint8)
        t = min(thickness + 30, min(h, w) // 4)
        
        mask[:t, :] = 255  # Top
        mask[-t:, :] = 255  # Bottom
        mask[:, :t] = 255  # Left
        mask[:, -t:] = 255  # Right
        
        # Save for Replicate
        img_buffer = BytesIO()
        img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        mask_img = Image.fromarray(mask)
        mask_buffer = BytesIO()
        mask_img.save(mask_buffer, format='PNG')
        mask_buffer.seek(0)
        
        # Try Replicate
        output = replicate.run(
            "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3",
            input={
                "image": img_buffer,
                "mask": mask_buffer,
                "prompt": "clean white background, product photography",
                "num_inference_steps": 20,
                "guidance_scale": 7.5
            }
        )
        
        if output:
            result_url = output[0] if isinstance(output, list) else output
            response = requests.get(result_url)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        
    except Exception as e:
        print(f"Replicate error: {str(e)}")
    
    # Fallback: aggressive crop
    crop_margin = thickness + 30
    return img.crop((crop_margin, crop_margin, img.width - crop_margin, img.height - crop_margin))


def create_thumbnail(img: Image.Image) -> Image.Image:
    """Create tight thumbnail"""
    # Simple edge-based crop
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Find non-white pixels
    _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    
    # Find bounding box
    coords = cv2.findNonZero(binary)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        
        # Add 3% padding
        pad_x = int(w * 0.03)
        pad_y = int(h * 0.03)
        
        x = max(0, x - pad_x)
        y = max(0, y - pad_y)
        w = min(img.width - x, w + 2 * pad_x)
        h = min(img.height - y, h + 2 * pad_y)
        
        img = img.crop((x, y, x + w, y + h))
    
    # Resize to target
    img = img.resize((1000, 1300), Image.Resampling.LANCZOS)
    
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.6)
    
    return img


def process_image(input_url: str) -> Dict[str, Any]:
    """Main processing pipeline"""
    try:
        # Load image
        img = load_image_from_url(input_url)
        
        # Convert to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Detect black frame
        detection = simple_black_frame_detection(img)
        
        # Remove if detected
        if detection['detected']:
            print(f"Black frame detected with thickness: {detection['thickness']}")
            img = remove_black_frame_simple(img, detection['thickness'])
        
        # Apply color correction
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.15)
        
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.12)
        
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(0.95)
        
        # Create thumbnail
        thumbnail = create_thumbnail(img)
        
        # Save
        buffer = BytesIO()
        thumbnail.save(buffer, format='JPEG', quality=95)
        buffer.seek(0)
        
        # Encode without padding
        thumbnail_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8').rstrip('=')
        
        return {
            "output": {
                "thumbnail": f"data:image/jpeg;base64,{thumbnail_base64}",
                "status": "success",
                "version": VERSION,
                "has_black_frame": detection['detected'],
                "inpainting_applied": detection['detected'],
                "frame_thickness": detection.get('thickness', 0),
                "message": "Thumbnail created successfully"
            }
        }
        
    except Exception as e:
        print(f"ERROR in process_image: {str(e)}")
        return {
            "output": {
                "status": "error",
                "error": f"Processing error: {str(e)}",
                "version": VERSION
            }
        }


# For RunPod
if __name__ == "__main__":
    print(f"Thumbnail Handler v{VERSION} loaded and ready")
