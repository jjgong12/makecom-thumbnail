import json
import base64
import traceback
import replicate
import requests
from io import BytesIO
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2

VERSION = "43"

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler for wedding ring thumbnail creation
    v43: Fixed masking detection and inpainting
    """
    print(f"Thumbnail Handler v{VERSION} starting...")
    
    thumbnail_handler = ThumbnailHandlerV43()
    
    try:
        # Find input URL with flexible search
        input_url = thumbnail_handler.find_input_url(event)
        
        if not input_url:
            print("ERROR: No input URL found")
            print(f"Event structure (first 500 chars): {str(event)[:500]}")
            return thumbnail_handler.create_error_response("No input URL found")
        
        # Process the image
        result = thumbnail_handler.process_image(input_url)
        return result
        
    except Exception as e:
        print(f"ERROR in handler: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return thumbnail_handler.create_error_response(f"Handler error: {str(e)}")


class ThumbnailHandlerV43:
    def __init__(self):
        self.version = VERSION
        self.thumbnail_size = (1000, 1300)
        
    def find_input_url(self, event: Dict[str, Any]) -> Optional[str]:
        """Find input URL from various possible locations"""
        # Priority keys to check
        keys_to_check = [
            'enhanced_image', 'image_url', 'url', 'image',
            'input_url', 'input_image', 'input'
        ]
        
        # Direct check
        for key in keys_to_check:
            if key in event and isinstance(event[key], str):
                url = event[key]
                if url.startswith(('http', 'data:')):
                    print(f"Found URL in key: {key}")
                    return url
        
        # Check input dict
        if 'input' in event and isinstance(event['input'], dict):
            for key in keys_to_check:
                if key in event['input'] and isinstance(event['input'][key], str):
                    url = event['input'][key]
                    if url.startswith(('http', 'data:')):
                        print(f"Found URL in input.{key}")
                        return url
        
        # Check numbered paths (like 4.data.output.output.enhanced_image)
        for i in range(10):
            path_parts = [
                [str(i), 'data', 'output', 'output', 'enhanced_image'],
                [str(i), 'data', 'output', 'enhanced_image'],
                [str(i), 'output', 'enhanced_image'],
                [str(i), 'enhanced_image']
            ]
            
            for parts in path_parts:
                try:
                    obj = event
                    for part in parts:
                        if isinstance(obj, dict) and part in obj:
                            obj = obj[part]
                        else:
                            break
                    else:
                        if isinstance(obj, str) and obj.startswith(('http', 'data:')):
                            print(f"Found URL at path: {'.'.join(parts)}")
                            return obj
                except:
                    continue
        
        # Deep search as last resort
        return self.find_url_recursive(event)
    
    def find_url_recursive(self, obj: Any, depth: int = 0, max_depth: int = 10) -> Optional[str]:
        """Recursively search for URL in nested structure"""
        if depth > max_depth:
            return None
        
        if isinstance(obj, str) and obj.startswith(('http', 'data:')):
            return obj
        elif isinstance(obj, dict):
            for value in obj.values():
                result = self.find_url_recursive(value, depth + 1, max_depth)
                if result:
                    return result
        elif isinstance(obj, list):
            for item in obj:
                result = self.find_url_recursive(item, depth + 1, max_depth)
                if result:
                    return result
        
        return None
    
    def load_image_from_url(self, url: str) -> Image.Image:
        """Load image from URL or data URL"""
        try:
            if url.startswith('data:'):
                # Handle data URL
                header, data = url.split(',', 1)
                # Add padding if needed
                padding = 4 - len(data) % 4
                if padding != 4:
                    data += '=' * padding
                img_data = base64.b64decode(data)
                return Image.open(BytesIO(img_data))
            else:
                # Handle regular URL
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
        """Detect black frame using multiple methods (based on v31 success)"""
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        results = []
        
        # Method 1: Ultra-sensitive black detection
        threshold = 25  # Very low threshold
        binary = (gray < threshold).astype(np.uint8) * 255
        
        # Check edges
        edge_thickness = 0
        for edge_name, edge_pixels in [
            ('top', binary[:100, :]),
            ('bottom', binary[-100:, :]),
            ('left', binary[:, :100]),
            ('right', binary[:, -100:])
        ]:
            if np.mean(edge_pixels) > 200:  # Mostly white (inverted)
                edge_thickness = max(edge_thickness, 100)
        
        if edge_thickness > 0:
            results.append({
                'method': 'ultra_sensitive',
                'detected': True,
                'thickness': edge_thickness
            })
        
        # Method 2: Contour-based detection
        _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            area = cw * ch
            
            # Check if it's a large rectangle touching edges
            if area > 0.3 * w * h:
                if x == 0 or y == 0 or x + cw == w or y + ch == h:
                    # Calculate thickness
                    thickness = min(x, y, w - (x + cw), h - (y + ch))
                    if thickness < 0:
                        thickness = 100  # Default if calculation fails
                    
                    results.append({
                        'method': 'contour',
                        'detected': True,
                        'thickness': thickness
                    })
                    break
        
        # Method 3: Multi-line edge scan
        scan_lines = 10
        edge_scores = {'top': [], 'bottom': [], 'left': [], 'right': []}
        
        for i in range(scan_lines):
            offset = i * 10
            
            # Top edge
            if np.mean(gray[offset, :]) < 30:
                edge_scores['top'].append(offset + 10)
            
            # Bottom edge
            if np.mean(gray[h - offset - 1, :]) < 30:
                edge_scores['bottom'].append(offset + 10)
            
            # Left edge
            if np.mean(gray[:, offset]) < 30:
                edge_scores['left'].append(offset + 10)
            
            # Right edge
            if np.mean(gray[:, w - offset - 1]) < 30:
                edge_scores['right'].append(offset + 10)
        
        # If multiple edges detected
        detected_edges = sum(1 for scores in edge_scores.values() if len(scores) > 3)
        if detected_edges >= 2:
            avg_thickness = sum(
                max(scores) for scores in edge_scores.values() if scores
            ) / max(1, sum(1 for scores in edge_scores.values() if scores))
            
            results.append({
                'method': 'multi_line',
                'detected': True,
                'thickness': int(avg_thickness)
            })
        
        # Method 4: Histogram analysis
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        dark_pixels = np.sum(hist[:30])
        total_pixels = w * h
        dark_ratio = dark_pixels / total_pixels
        
        if dark_ratio > 0.1:  # More than 10% very dark pixels
            # Estimate thickness based on ratio
            estimated_thickness = int(min(w, h) * dark_ratio / 4)
            results.append({
                'method': 'histogram',
                'detected': True,
                'thickness': estimated_thickness
            })
        
        # Combine results
        if len(results) >= 2:  # At least 2 methods agree
            avg_thickness = sum(r['thickness'] for r in results) / len(results)
            return {
                'detected': True,
                'thickness': int(avg_thickness),
                'confidence': len(results) / 4.0,
                'methods': [r['method'] for r in results]
            }
        
        return {
            'detected': False,
            'thickness': 0,
            'confidence': 0,
            'methods': []
        }
    
    def remove_black_frame_with_replicate(self, img: Image.Image, thickness: int) -> Image.Image:
        """Remove black frame using Replicate API inpainting"""
        try:
            print(f"Removing black frame with thickness: {thickness}")
            
            # Create mask for inpainting
            img_np = np.array(img)
            h, w = img_np.shape[:2]
            
            # Create mask (white where we want to inpaint)
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # Mark edges for inpainting (with some expansion)
            expand = 10  # Expand mask slightly
            t = thickness + expand
            
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
                    "prompt": "clean white background, professional product photography background, bright studio lighting",
                    "negative_prompt": "black frame, black border, dark edges, shadows, vignette",
                    "num_inference_steps": 25,
                    "guidance_scale": 7.5,
                    "scheduler": "K_EULER"
                }
            )
            
            # Get result
            if output and len(output) > 0:
                result_url = output[0] if isinstance(output, list) else output
                response = requests.get(result_url)
                response.raise_for_status()
                
                inpainted = Image.open(BytesIO(response.content))
                
                # Verify black frame is gone
                check = self.detect_black_frame_multi_method(inpainted)
                if not check['detected']:
                    print("Black frame successfully removed")
                    return inpainted
                else:
                    print("Black frame still detected after inpainting, using crop fallback")
            
        except Exception as e:
            print(f"Replicate inpainting error: {str(e)}")
        
        # Fallback: aggressive crop
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
        """Apply color correction after masking removal"""
        # Brightness
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.15)
        
        # Contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.12)
        
        # Saturation
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(0.95)
        
        return img
    
    def create_tight_thumbnail(self, img: Image.Image) -> Image.Image:
        """Create tight thumbnail with ring centered"""
        # Find ring bounds
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Use edge detection to find ring
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get bounding box of all contours
            x_min, y_min = img.width, img.height
            x_max, y_max = 0, 0
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w)
                y_max = max(y_max, y + h)
            
            # Add small padding (3%)
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
        
        # Apply detail enhancement
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.6)  # Strong sharpness for thumbnails
        
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
                # Apply color correction after masking removal
                img = self.apply_color_correction(img)
            else:
                print("No black frame detected")
                # Apply color correction directly
                img = self.apply_color_correction(img)
            
            # Create tight thumbnail
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
    print(f"Thumbnail Handler v{VERSION} loaded successfully
