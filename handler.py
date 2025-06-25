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

VERSION = "44"

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler for wedding ring thumbnail creation
    v44: Ultra sensitive masking detection
    """
    print(f"Thumbnail Handler v{VERSION} starting...")
    print(f"Event type: {type(event)}")
    print(f"Event keys: {list(event.keys()) if isinstance(event, dict) else 'Not a dict'}")
    
    thumbnail_handler = ThumbnailHandlerV44()
    
    try:
        # Find input URL with flexible search
        input_url = thumbnail_handler.find_input_url(event)
        
        if not input_url:
            print("ERROR: No input URL found")
            print(f"Full event (first 1000 chars): {str(event)[:1000]}")
            return thumbnail_handler.create_error_response("No input URL found. Please check the input structure.")
        
        print(f"Found input URL, type: {'data URL' if input_url.startswith('data:') else 'HTTP URL'}")
        
        # Process the image
        result = thumbnail_handler.process_image(input_url)
        
        print(f"Result status: {result.get('output', {}).get('status')}")
        return result
        
    except Exception as e:
        print(f"ERROR in handler: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return thumbnail_handler.create_error_response(f"Handler error: {str(e)}")


class ThumbnailHandlerV44:
    def __init__(self):
        self.version = VERSION
        self.thumbnail_size = (1000, 1300)
        
    def find_input_url(self, event: Dict[str, Any]) -> Optional[str]:
        """Find input URL from various possible locations"""
        # Priority keys to check
        keys_to_check = [
            'enhanced_image', 'image_url', 'url', 'image',
            'input_url', 'input_image', 'input', 'enhancedImage',
            'imageUrl', 'img'
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
                # Remove any whitespace
                data = ''.join(data.split())
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
    
    def detect_black_frame_ultra_sensitive(self, img: Image.Image) -> Dict[str, Any]:
        """Ultra sensitive black frame detection"""
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        print(f"Image size for detection: {w}x{h}")
        
        results = []
        
        # Method 1: Super Ultra-sensitive black detection
        for threshold in [15, 20, 25, 30, 35, 40]:  # Start from very low threshold
            binary = (gray < threshold).astype(np.uint8) * 255
            
            # Check all edges with varying depths
            for check_depth in [50, 100, 150, 200]:
                if check_depth > min(h, w) // 4:
                    continue
                
                # Check each edge
                edge_black = {
                    'top': np.mean(binary[:check_depth, :]) > 200,
                    'bottom': np.mean(binary[-check_depth:, :]) > 200,
                    'left': np.mean(binary[:, :check_depth]) > 200,
                    'right': np.mean(binary[:, -check_depth:]) > 200
                }
                
                # If at least 3 edges are black
                if sum(edge_black.values()) >= 3:
                    results.append({
                        'method': f'ultra_sensitive_t{threshold}_d{check_depth}',
                        'detected': True,
                        'thickness': check_depth
                    })
                    print(f"Detected with threshold {threshold}, depth {check_depth}")
                    break
        
        # Method 2: Gradient detection for sharp edges
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        
        # Look for strong gradients at edges
        edge_gradient_strength = {
            'top': np.mean(gradient[:50, :]),
            'bottom': np.mean(gradient[-50:, :]),
            'left': np.mean(gradient[:, :50]),
            'right': np.mean(gradient[:, -50:])
        }
        
        strong_edges = sum(1 for v in edge_gradient_strength.values() if v > 50)
        if strong_edges >= 3:
            results.append({
                'method': 'gradient_detection',
                'detected': True,
                'thickness': 100  # Estimate
            })
            print("Detected with gradient method")
        
        # Method 3: Corner darkness check
        corner_size = 200
        corners = [
            gray[:corner_size, :corner_size],  # Top-left
            gray[:corner_size, -corner_size:],  # Top-right
            gray[-corner_size:, :corner_size],  # Bottom-left
            gray[-corner_size:, -corner_size:]  # Bottom-right
        ]
        
        dark_corners = sum(1 for corner in corners if np.mean(corner) < 40)
        if dark_corners >= 3:
            results.append({
                'method': 'corner_darkness',
                'detected': True,
                'thickness': 150  # Estimate
            })
            print("Detected with corner darkness method")
        
        # Method 4: Morphological analysis
        kernel = np.ones((5, 5), np.uint8)
        _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find large black regions
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(255 - closed, connectivity=8)
        
        for i in range(1, num_labels):
            x, y, width, height, area = stats[i]
            # Check if it's a frame-like structure
            if area > 0.2 * w * h:  # Large area
                if x == 0 or y == 0 or x + width == w or y + height == h:  # Touches edge
                    thickness = min(x, y, w - (x + width), h - (y + height))
                    if thickness < 0:
                        thickness = 100
                    results.append({
                        'method': 'morphological',
                        'detected': True,
                        'thickness': abs(thickness)
                    })
                    print("Detected with morphological method")
                    break
        
        # If ANY method detected a frame, return positive
        if results:
            avg_thickness = int(np.mean([r['thickness'] for r in results]))
            print(f"Black frame DETECTED by {len(results)} methods, avg thickness: {avg_thickness}")
            return {
                'detected': True,
                'thickness': avg_thickness,
                'confidence': len(results) / 4.0,
                'methods': [r['method'] for r in results]
            }
        
        print("No black frame detected by any method")
        return {
            'detected': False,
            'thickness': 0,
            'confidence': 0,
            'methods': []
        }
    
    def remove_black_frame_with_replicate(self, img: Image.Image, thickness: int) -> Image.Image:
        """Remove black frame using Replicate API inpainting"""
        try:
            print(f"Starting Replicate inpainting for black frame removal, thickness: {thickness}")
            
            # Create mask for inpainting
            img_np = np.array(img)
            h, w = img_np.shape[:2]
            
            # Create mask (white where we want to inpaint)
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # Expand thickness for better results
            t = min(thickness + 30, min(h, w) // 4)  # Don't take more than 1/4 of image
            
            # Create frame mask
            mask[:t, :] = 255  # Top
            mask[-t:, :] = 255  # Bottom
            mask[:, :t] = 255  # Left
            mask[:, -t:] = 255  # Right
            
            print(f"Mask created with expanded thickness: {t}")
            
            # Save images for Replicate
            img_buffer = BytesIO()
            img.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            mask_img = Image.fromarray(mask)
            mask_buffer = BytesIO()
            mask_img.save(mask_buffer, format='PNG')
            mask_buffer.seek(0)
            
            print("Calling Replicate API...")
            
            # Run inpainting with Replicate
            output = replicate.run(
                "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3",
                input={
                    "image": img_buffer,
                    "mask": mask_buffer,
                    "prompt": "clean white background, professional product photography background, pure white studio background, bright even lighting",
                    "negative_prompt": "black frame, black border, dark edges, shadows, vignette, darkness, gray areas",
                    "num_inference_steps": 30,  # Increased for better quality
                    "guidance_scale": 8.5,
                    "scheduler": "K_EULER_ANCESTRAL"
                }
            )
            
            # Get result
            if output and len(output) > 0:
                result_url = output[0] if isinstance(output, list) else output
                print(f"Replicate returned URL: {result_url}")
                
                response = requests.get(result_url)
                response.raise_for_status()
                
                inpainted = Image.open(BytesIO(response.content))
                print("Inpainting successful, checking results...")
                
                # Verify black frame is gone
                check = self.detect_black_frame_ultra_sensitive(inpainted)
                if not check['detected']:
                    print("Black frame successfully removed!")
                    return inpainted
                else:
                    print("Black frame still detected after inpainting, using aggressive crop")
            else:
                print("No output from Replicate")
            
        except Exception as e:
            print(f"Replicate inpainting error: {str(e)}")
            print(f"Error type: {type(e).__name__}")
        
        # Fallback: very aggressive crop
        print("Using fallback aggressive crop")
        return self.crop_black_frame(img, thickness + 50)
    
    def crop_black_frame(self, img: Image.Image, thickness: int) -> Image.Image:
        """Fallback: aggressively crop out black frame"""
        width, height = img.size
        # Add extra margin to ensure complete removal
        crop_margin = thickness + 20
        crop_box = (
            crop_margin,
            crop_margin,
            width - crop_margin,
            height - crop_margin
        )
        
        # Ensure valid crop box
        if crop_box[2] > crop_box[0] and crop_box[3] > crop_box[1]:
            return img.crop(crop_box)
        else:
            # If crop would be invalid, return center portion
            center_crop = min(width, height) // 2
            crop_box = (
                width // 2 - center_crop // 2,
                height // 2 - center_crop // 2,
                width // 2 + center_crop // 2,
                height // 2 + center_crop // 2
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
        edges = cv2.Canny(gray, 30, 100)  # Lower thresholds for better detection
        
        # Dilate to connect edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get bounding box of all significant contours
            x_min, y_min = img.width, img.height
            x_max, y_max = 0, 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Filter out tiny contours
                    x, y, w, h = cv2.boundingRect(contour)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x + w)
                    y_max = max(y_max, y + h)
            
            # Add very small padding (3%)
            padding = 0.03
            pad_x = int((x_max - x_min) * padding)
            pad_y = int((y_max - y_min) * padding)
            
            x_min = max(0, x_min - pad_x)
            y_min = max(0, y_min - pad_y)
            x_max = min(img.width, x_max + pad_x)
            y_max = min(img.height, y_max + pad_y)
            
            # Crop
            if x_max > x_min and y_max > y_min:
                img = img.crop((x_min, y_min, x_max, y_max))
        
        # Resize to target size
        img = img.resize(self.thumbnail_size, Image.Resampling.LANCZOS)
        
        # Apply strong detail enhancement
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.6)  # Strong sharpness for thumbnails
        
        return img
    
    def process_image(self, input_url: str) -> Dict[str, Any]:
        """Main processing pipeline"""
        try:
            # Load image
            print("Loading image...")
            img = self.load_image_from_url(input_url)
            print(f"Image loaded: {img.size}, mode: {img.mode}")
            
            # Convert to RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Detect black frame with ultra-sensitive detection
            print("Detecting black frame...")
            detection = self.detect_black_frame_ultra_sensitive(img)
            
            # Remove black frame if detected
            if detection['detected']:
                print(f"Black frame detected with confidence {detection['confidence']}: {detection}")
                img = self.remove_black_frame_with_replicate(img, detection['thickness'])
                # Apply color correction after masking removal
                img = self.apply_color_correction(img)
            else:
                print("No black frame detected, applying color correction directly")
                # Apply color correction directly
                img = self.apply_color_correction(img)
            
            # Create tight thumbnail
            print("Creating thumbnail...")
            thumbnail = self.create_tight_thumbnail(img)
            
            # Save to buffer
            buffer = BytesIO()
            thumbnail.save(buffer, format='JPEG', quality=95, optimize=True)
            buffer.seek(0)
            
            # Encode without padding for Make.com
            thumbnail_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8').rstrip('=')
            print(f"Thumbnail base64 length: {len(thumbnail_base64)}")
            
            # Return with correct structure
            result = {
                "output": {
                    "thumbnail": f"data:image/jpeg;base64,{thumbnail_base64}",
                    "status": "success",
                    "version": self.version,
                    "has_black_frame": detection['detected'],
                    "inpainting_applied": detection['detected'],
                    "mask_created": detection['detected'],
                    "frame_thickness": detection.get('thickness', 0),
                    "detection_confidence": detection.get('confidence', 0),
                    "detection_methods": detection.get('methods', []),
                    "thumbnail_size": f"{self.thumbnail_size[0]}x{self.thumbnail_size[1]}",
                    "message": "Thumbnail created successfully"
                }
            }
            
            print("Returning success response")
            return result
            
        except Exception as e:
            print(f"ERROR in process_image: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
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
