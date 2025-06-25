import json
import base64
import traceback
import replicate
import requests
from io import BytesIO
from typing import Dict, Any, Optional, Tuple
from PIL import Image, ImageEnhance
import numpy as np
import cv2

VERSION = "46"

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler for wedding ring thumbnail creation
    v46: Direct path checking without recursion (based on v41 approach)
    """
    print(f"Thumbnail Handler v{VERSION} starting...")
    
    thumbnail_handler = ThumbnailHandlerV46()
    
    try:
        # Find input URL using direct checks only
        input_url = thumbnail_handler.find_input_url(event)
        
        if not input_url:
            print("ERROR: No input URL found")
            print(f"Event keys: {list(event.keys()) if isinstance(event, dict) else 'Not a dict'}")
            print(f"Event structure (first 500 chars): {str(event)[:500]}")
            return thumbnail_handler.create_error_response("No input URL found. Please check the input structure.")
        
        print(f"Found input URL, type: {'data URL' if input_url.startswith('data:') else 'HTTP URL'}")
        
        # Process the image
        result = thumbnail_handler.process_image(input_url)
        return result
        
    except Exception as e:
        print(f"ERROR in handler: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return thumbnail_handler.create_error_response(f"Handler error: {str(e)}")


class ThumbnailHandlerV46:
    def __init__(self):
        self.version = VERSION
        self.thumbnail_size = (1000, 1300)
    
    def find_input_url(self, event: Dict[str, Any]) -> Optional[str]:
        """Find input URL using direct path checks only - NO RECURSION"""
        # Priority keys to check
        url_keys = ['enhanced_image', 'image_url', 'url', 'image', 'input_url', 
                    'input_image', 'enhancedImage', 'imageUrl', 'img']
        
        # Check direct keys
        for key in url_keys:
            if key in event and isinstance(event[key], str):
                url = event[key]
                if url.startswith(('http', 'data:')):
                    print(f"Found URL in direct key: {key}")
                    return url
        
        # Check input dict
        if 'input' in event and isinstance(event['input'], dict):
            for key in url_keys:
                if key in event['input'] and isinstance(event['input'][key], str):
                    url = event['input'][key]
                    if url.startswith(('http', 'data:')):
                        print(f"Found URL in input.{key}")
                        return url
        
        # Check numbered keys with common patterns
        for i in range(10):
            str_i = str(i)
            
            # Pattern: {i}.data.output.output.enhanced_image
            if str_i in event and isinstance(event[str_i], dict):
                current = event[str_i]
                
                # Check common patterns
                patterns = [
                    ['data', 'output', 'output', 'enhanced_image'],
                    ['data', 'output', 'enhanced_image'],
                    ['output', 'enhanced_image'],
                    ['enhanced_image']
                ]
                
                for pattern in patterns:
                    temp = current
                    valid = True
                    
                    for part in pattern:
                        if isinstance(temp, dict) and part in temp:
                            temp = temp[part]
                        else:
                            valid = False
                            break
                    
                    if valid and isinstance(temp, str) and temp.startswith(('http', 'data:')):
                        print(f"Found URL at path: {str_i}.{'.'.join(pattern)}")
                        return temp
        
        # Check 'data' key patterns
        if 'data' in event and isinstance(event['data'], dict):
            data = event['data']
            
            # Check direct keys in data
            for key in url_keys:
                if key in data and isinstance(data[key], str):
                    url = data[key]
                    if url.startswith(('http', 'data:')):
                        print(f"Found URL in data.{key}")
                        return url
            
            # Check data.output patterns
            if 'output' in data and isinstance(data['output'], dict):
                output = data['output']
                
                # Check direct keys in output
                for key in url_keys:
                    if key in output and isinstance(output[key], str):
                        url = output[key]
                        if url.startswith(('http', 'data:')):
                            print(f"Found URL in data.output.{key}")
                            return url
                
                # Check data.output.output pattern
                if 'output' in output and isinstance(output['output'], dict):
                    output2 = output['output']
                    for key in url_keys:
                        if key in output2 and isinstance(output2[key], str):
                            url = output2[key]
                            if url.startswith(('http', 'data:')):
                                print(f"Found URL in data.output.output.{key}")
                                return url
        
        print("No URL found in any known location")
        return None
    
    def load_image_from_url(self, url: str) -> Image.Image:
        """Load image from URL or data URL"""
        try:
            if url.startswith('data:'):
                # Handle data URL
                header, data = url.split(',', 1)
                # Remove any whitespace
                data = ''.join(data.split())
                # Add padding if needed
                padding = 4 - len(data) % 4
                if padding != 4:
                    data += '=' * padding
                img_data = base64.b64decode(data)
                return Image.open(BytesIO(img_data))
            else:
                # Handle regular URL with multiple attempts
                headers_list = [
                    {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'},
                    {'User-Agent': 'Python-Requests/2.31.0'},
                    {}
                ]
                
                for headers in headers_list:
                    try:
                        response = requests.get(url, headers=headers, timeout=30, stream=True)
                        response.raise_for_status()
                        return Image.open(BytesIO(response.content))
                    except Exception as e:
                        print(f"Failed with headers {headers}: {str(e)}")
                        continue
                
                raise ValueError("Failed to load image with all header attempts")
                
        except Exception as e:
            print(f"ERROR loading image: {str(e)}")
            raise
    
    def detect_black_frame_multi_method(self, img: Image.Image) -> Dict[str, Any]:
        """Detect black frame using multiple methods"""
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        results = []
        
        # Method 1: Ultra-sensitive black detection
        threshold = 25
        binary = (gray < threshold).astype(np.uint8) * 255
        
        # Check edges
        edge_thickness = 0
        check_depth = min(100, min(h, w) // 10)
        
        if np.mean(binary[:check_depth, :]) > 200:  # Top
            edge_thickness = check_depth
        if np.mean(binary[-check_depth:, :]) > 200:  # Bottom
            edge_thickness = max(edge_thickness, check_depth)
        if np.mean(binary[:, :check_depth]) > 200:  # Left
            edge_thickness = max(edge_thickness, check_depth)
        if np.mean(binary[:, -check_depth:]) > 200:  # Right
            edge_thickness = max(edge_thickness, check_depth)
        
        if edge_thickness > 0:
            results.append({
                'method': 'ultra_sensitive',
                'detected': True,
                'thickness': edge_thickness
            })
        
        # Method 2: Multi-line edge scan
        scan_lines = 10
        dark_edges = 0
        
        for i in range(scan_lines):
            offset = i * 10
            if offset >= min(h, w) // 4:
                break
            
            # Check all edges
            if np.mean(gray[offset, :]) < 30:  # Top
                dark_edges += 1
            if np.mean(gray[h - offset - 1, :]) < 30:  # Bottom
                dark_edges += 1
            if np.mean(gray[:, offset]) < 30:  # Left
                dark_edges += 1
            if np.mean(gray[:, w - offset - 1]) < 30:  # Right
                dark_edges += 1
        
        if dark_edges >= scan_lines * 3:  # At least 3 edges dark
            results.append({
                'method': 'multi_line',
                'detected': True,
                'thickness': 100
            })
        
        # If any method detected a frame
        if results:
            avg_thickness = int(np.mean([r['thickness'] for r in results]))
            return {
                'detected': True,
                'thickness': avg_thickness,
                'methods': [r['method'] for r in results]
            }
        
        return {
            'detected': False,
            'thickness': 0,
            'methods': []
        }
    
    def remove_black_frame_with_replicate(self, img: Image.Image, thickness: int) -> Image.Image:
        """Remove black frame using Replicate API inpainting"""
        try:
            print(f"Removing black frame with thickness: {thickness}")
            
            # Create mask for inpainting
            img_np = np.array(img)
            h, w = img_np.shape[:2]
            
            # Create mask
            mask = np.zeros((h, w), dtype=np.uint8)
            t = min(thickness + 20, min(h, w) // 4)
            
            mask[:t, :] = 255  # Top
            mask[-t:, :] = 255  # Bottom
            mask[:, :t] = 255  # Left
            mask[:, -t:] = 255  # Right
            
            # Save images for Replicate
            img_buffer = BytesIO()
            img.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            mask_img = Image.fromarray(mask)
            mask_buffer = BytesIO()
            mask_img.save(mask_buffer, format='PNG')
            mask_buffer.seek(0)
            
            # Run inpainting
            output = replicate.run(
                "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3",
                input={
                    "image": img_buffer,
                    "mask": mask_buffer,
                    "prompt": "clean white background, professional product photography background",
                    "negative_prompt": "black frame, black border, dark edges",
                    "num_inference_steps": 25,
                    "guidance_scale": 7.5
                }
            )
            
            # Get result
            if output and len(output) > 0:
                result_url = output[0] if isinstance(output, list) else output
                response = requests.get(result_url)
                response.raise_for_status()
                
                inpainted = Image.open(BytesIO(response.content))
                print("Black frame removed with inpainting")
                return inpainted
            
        except Exception as e:
            print(f"Replicate inpainting error: {str(e)}")
        
        # Fallback: crop
        return self.crop_black_frame(img, thickness + 20)
    
    def crop_black_frame(self, img: Image.Image, thickness: int) -> Image.Image:
        """Fallback: crop out black frame"""
        width, height = img.size
        crop_box = (
            thickness,
            thickness,
            width - thickness,
            height - thickness
        )
        return img.crop(crop_box)
    
    def apply_color_correction(self, img: Image.Image) -> Image.Image:
        """Apply color correction"""
        # Brightness
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.15)
        
        # Contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.12)
        
        # Color
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(0.95)
        
        return img
    
    def create_tight_thumbnail(self, img: Image.Image) -> Image.Image:
        """Create tight thumbnail with ring centered"""
        # Find ring bounds
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Use edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get bounding box
            x_min, y_min = img.width, img.height
            x_max, y_max = 0, 0
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w)
                y_max = max(y_max, y + h)
            
            # Add 3% padding
            padding = 0.03
            pad_x = int((x_max - x_min) * padding)
            pad_y = int((y_max - y_min) * padding)
            
            x_min = max(0, x_min - pad_x)
            y_min = max(0, y_min - pad_y)
            x_max = min(img.width, x_max + pad_x)
            y_max = min(img.height, y_max + pad_y)
            
            # Crop
            img = img.crop((x_min, y_min, x_max, y_max))
        
        # Resize to target size
        img = img.resize(self.thumbnail_size, Image.Resampling.LANCZOS)
        
        # Apply sharpness
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.6)
        
        return img
    
    def process_image(self, input_url: str) -> Dict[str, Any]:
        """Main processing pipeline"""
        try:
            # Load image
            img = self.load_image_from_url(input_url)
            
            # Convert to RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Detect black frame
            detection = self.detect_black_frame_multi_method(img)
            
            # Remove black frame if detected
            if detection['detected']:
                print(f"Black frame detected: {detection}")
                img = self.remove_black_frame_with_replicate(img, detection['thickness'])
            
            # Apply color correction
            img = self.apply_color_correction(img)
            
            # Create thumbnail
            thumbnail = self.create_tight_thumbnail(img)
            
            # Save to buffer
            buffer = BytesIO()
            thumbnail.save(buffer, format='JPEG', quality=95, optimize=True)
            buffer.seek(0)
            
            # Encode without padding
            thumbnail_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8').rstrip('=')
            
            # Return with correct structure
            return {
                "output": {
                    "thumbnail": f"data:image/jpeg;base64,{thumbnail_base64}",
                    "status": "success",
                    "version": self.version,
                    "has_black_frame": detection['detected'],
                    "inpainting_applied": detection['detected'],
                    "mask_created": detection['detected'],
                    "frame_thickness": detection.get('thickness', 0),
                    "detection_methods": detection.get('methods', []),
                    "thumbnail_size": f"{self.thumbnail_size[0]}x{self.thumbnail_size[1]}",
                    "message": "Thumbnail created successfully"
                }
            }
            
        except Exception as e:
            print(f"ERROR in process_image: {str(e)}")
            return self.create_error_response(f"Processing error: {str(e)}")
    
    def create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create error response with correct structure"""
        return {
            "output": {
                "status": "error",
                "error": error_message,
                "version": self.version
            }
        }


# For RunPod
if __name__ == "__main__":
    print(f"Thumbnail Handler v{VERSION} loaded successfully")
