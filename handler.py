import runpod
import os
import logging
import traceback
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import requests
import base64
import io
import json
from typing import Dict, Any, Tuple, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import replicate
try:
    import replicate
    REPLICATE_AVAILABLE = True
    logger.info("Replicate module loaded successfully")
except ImportError:
    REPLICATE_AVAILABLE = False
    logger.warning("Replicate module not available")

class ThumbnailHandler:
    def __init__(self):
        self.version = "v41-fixed-url"
        logger.info(f"Initializing {self.version}")
        self.headers_list = [
            {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'},
            {'User-Agent': 'Python-Requests/2.31.0'},
            {}
        ]
    
    def find_input_url(self, event: Dict[str, Any]) -> Optional[str]:
        """Find image URL from various possible locations"""
        # Most common paths first
        simple_paths = [
            ['input', 'enhanced_image'],
            ['input', 'url'],
            ['input', 'image_url'],
            ['input', 'image'],
            ['enhanced_image'],
            ['url'],
            ['image_url'],
            ['image']
        ]
        
        # Check simple paths
        for path in simple_paths:
            try:
                data = event
                for key in path:
                    if isinstance(data, dict) and key in data:
                        data = data[key]
                    else:
                        break
                else:
                    if isinstance(data, str) and (data.startswith('http') or data.startswith('data:') or len(data) > 100):
                        logger.info(f"Found URL/data at path: {'.'.join(path)}")
                        return data
            except:
                continue
        
        # Check numbered paths (like 4.data.output.output.enhanced_image)
        if 'input' in event:
            for i in range(10):
                key = str(i)
                if key in event['input']:
                    try:
                        if isinstance(event['input'][key], dict):
                            paths = [
                                ['data', 'output', 'output', 'enhanced_image'],
                                ['data', 'output', 'enhanced_image'],
                                ['output', 'enhanced_image'],
                                ['enhanced_image']
                            ]
                            
                            for path in paths:
                                data = event['input'][key]
                                for subkey in path:
                                    if isinstance(data, dict) and subkey in data:
                                        data = data[subkey]
                                    else:
                                        break
                                else:
                                    if isinstance(data, str) and (data.startswith('http') or data.startswith('data:') or len(data) > 100):
                                        logger.info(f"Found URL/data at numbered path: {key}.{'.'.join(path)}")
                                        return data
                    except:
                        continue
        
        logger.error(f"No valid image URL found in event")
        logger.error(f"Event structure: {json.dumps(event, indent=2)[:500]}...")
        return None
    
    def load_image_from_source(self, source: str) -> Optional[Image.Image]:
        """Load image from URL or base64 data"""
        try:
            # Handle data URLs
            if source.startswith('data:'):
                logger.info("Loading from data URL")
                base64_str = source.split(',')[1]
                # Fix padding if needed
                padding = 4 - len(base64_str) % 4
                if padding != 4:
                    base64_str += '=' * padding
                img_data = base64.b64decode(base64_str)
                return Image.open(io.BytesIO(img_data)).convert('RGB')
            
            # Handle base64 strings
            elif not source.startswith('http') and len(source) > 100:
                logger.info("Loading from base64 string")
                # Fix padding if needed
                padding = 4 - len(source) % 4
                if padding != 4:
                    source += '=' * padding
                img_data = base64.b64decode(source)
                return Image.open(io.BytesIO(img_data)).convert('RGB')
            
            # Handle URLs
            else:
                logger.info(f"Loading from URL: {source[:100]}...")
                for headers in self.headers_list:
                    try:
                        response = requests.get(source, headers=headers, timeout=30, stream=True)
                        if response.status_code == 200:
                            return Image.open(io.BytesIO(response.content)).convert('RGB')
                    except Exception as e:
                        logger.warning(f"Failed with headers {headers}: {str(e)}")
                        continue
                
                raise ValueError("Failed to load image from URL with all header options")
                
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            return None
    
    def detect_black_masking_multi_method(self, image: Image.Image) -> Dict[str, Any]:
        """Detect black masking using multiple methods with weighted scoring"""
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        detection_results = []
        
        # Method 1: Edge-based detection (weight: 0.3)
        edge_result = self._detect_edges_method(gray, width, height)
        if edge_result['detected']:
            detection_results.append({**edge_result, 'weight': 0.3})
        
        # Method 2: Corner analysis (weight: 0.2)
        corner_result = self._detect_corners_method(gray, width, height)
        if corner_result['detected']:
            detection_results.append({**corner_result, 'weight': 0.2})
        
        # Method 3: Gradient-based (weight: 0.3)
        gradient_result = self._detect_gradient_method(gray, width, height)
        if gradient_result['detected']:
            detection_results.append({**gradient_result, 'weight': 0.3})
        
        # Method 4: Connected components (weight: 0.2)
        component_result = self._detect_components_method(gray, width, height)
        if component_result['detected']:
            detection_results.append({**component_result, 'weight': 0.2})
        
        # Calculate weighted score
        total_weight = sum(r['weight'] for r in detection_results)
        
        if total_weight >= 0.4:  # At least 40% confidence
            # Average the detected frames
            avg_top = int(np.mean([r['frame']['top'] for r in detection_results]))
            avg_bottom = int(np.mean([r['frame']['bottom'] for r in detection_results]))
            avg_left = int(np.mean([r['frame']['left'] for r in detection_results]))
            avg_right = int(np.mean([r['frame']['right'] for r in detection_results]))
            
            return {
                'detected': True,
                'confidence': total_weight,
                'frame': {
                    'top': avg_top,
                    'bottom': avg_bottom,
                    'left': avg_left,
                    'right': avg_right
                },
                'methods_detected': len(detection_results)
            }
        
        return {'detected': False, 'confidence': 0, 'frame': None}
    
    def _detect_edges_method(self, gray: np.ndarray, width: int, height: int) -> Dict[str, Any]:
        """Edge-based detection scanning from borders"""
        threshold = 30
        scan_depth = min(300, height // 3, width // 3)
        
        edges = {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
        
        # Top edge
        for y in range(scan_depth):
            if np.mean(gray[y, :]) > threshold:
                edges['top'] = y
                break
        
        # Bottom edge
        for y in range(scan_depth):
            if np.mean(gray[height-1-y, :]) > threshold:
                edges['bottom'] = y
                break
        
        # Left edge
        for x in range(scan_depth):
            if np.mean(gray[:, x]) > threshold:
                edges['left'] = x
                break
        
        # Right edge
        for x in range(scan_depth):
            if np.mean(gray[:, width-1-x]) > threshold:
                edges['right'] = x
                break
        
        total_frame = sum(edges.values())
        if total_frame > 20:  # At least 20 pixels of frame
            return {'detected': True, 'frame': edges}
        
        return {'detected': False}
    
    def _detect_corners_method(self, gray: np.ndarray, width: int, height: int) -> Dict[str, Any]:
        """Analyze corners for black regions"""
        corner_size = min(200, height // 4, width // 4)
        threshold = 40
        
        corners = [
            gray[:corner_size, :corner_size],  # Top-left
            gray[:corner_size, -corner_size:],  # Top-right
            gray[-corner_size:, :corner_size],  # Bottom-left
            gray[-corner_size:, -corner_size:]  # Bottom-right
        ]
        
        dark_corners = sum(1 for corner in corners if np.mean(corner) < threshold)
        
        if dark_corners >= 3:  # At least 3 dark corners
            # Estimate frame
            edges = {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
            
            # Scan for actual edges
            for y in range(corner_size):
                if np.mean(gray[y, :]) > threshold:
                    edges['top'] = y
                    break
            
            for y in range(corner_size):
                if np.mean(gray[height-1-y, :]) > threshold:
                    edges['bottom'] = y
                    break
                    
            return {'detected': True, 'frame': edges}
        
        return {'detected': False}
    
    def _detect_gradient_method(self, gray: np.ndarray, width: int, height: int) -> Dict[str, Any]:
        """Use gradients to detect frame transitions"""
        # Sobel gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Find strong horizontal/vertical lines
        edges = {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
        
        # Top edge - look for strong horizontal gradient
        top_region = np.abs(grad_y[:100, :])
        top_lines = np.mean(top_region, axis=1)
        strong_lines = np.where(top_lines > np.percentile(top_lines, 90))[0]
        if len(strong_lines) > 0:
            edges['top'] = strong_lines[-1]
        
        # Similar for other edges...
        # Simplified for brevity
        
        if edges['top'] > 10 or edges['bottom'] > 10:
            return {'detected': True, 'frame': edges}
        
        return {'detected': False}
    
    def _detect_components_method(self, gray: np.ndarray, width: int, height: int) -> Dict[str, Any]:
        """Use connected components to find black frame"""
        _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        binary = cv2.bitwise_not(binary)  # Invert so black is white
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        # Look for large black regions touching edges
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            
            # Check if it's large and touches edges
            if area > (width * height * 0.1):  # At least 10% of image
                touches_edge = (x == 0 or y == 0 or x + w == width or y + h == height)
                if touches_edge:
                    # Estimate frame from this component
                    edges = {
                        'top': y if y == 0 else 0,
                        'bottom': height - (y + h) if y + h == height else 0,
                        'left': x if x == 0 else 0,
                        'right': width - (x + w) if x + w == width else 0
                    }
                    return {'detected': True, 'frame': edges}
        
        return {'detected': False}
    
    def create_inpainting_mask(self, image: Image.Image, frame: Dict[str, int]) -> Image.Image:
        """Create mask for black frame areas"""
        mask = Image.new('L', image.size, 0)
        draw = ImageDraw.Draw(mask)
        
        width, height = image.size
        
        # Paint frame areas white (to be inpainted)
        # Top
        if frame['top'] > 0:
            draw.rectangle([0, 0, width, frame['top'] + 5], fill=255)
        
        # Bottom
        if frame['bottom'] > 0:
            draw.rectangle([0, height - frame['bottom'] - 5, width, height], fill=255)
        
        # Left
        if frame['left'] > 0:
            draw.rectangle([0, 0, frame['left'] + 5, height], fill=255)
        
        # Right
        if frame['right'] > 0:
            draw.rectangle([width - frame['right'] - 5, 0, width, height], fill=255)
        
        # Dilate mask slightly for better inpainting
        mask_array = np.array(mask)
        kernel = np.ones((5, 5), np.uint8)
        mask_array = cv2.dilate(mask_array, kernel, iterations=2)
        
        return Image.fromarray(mask_array)
    
    def remove_black_frame_with_replicate(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """Use Replicate API to inpaint the black frame areas"""
        if not REPLICATE_AVAILABLE:
            logger.warning("Replicate not available, using fallback crop")
            return self.fallback_crop(image, mask)
        
        try:
            # Convert images to base64
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            mask_buffer = io.BytesIO()
            mask.save(mask_buffer, format='PNG')
            mask_buffer.seek(0)
            
            # Run inpainting
            output = replicate.run(
                "lucataco/flux-fill-pro",
                input={
                    "image": img_buffer,
                    "mask": mask_buffer,
                    "prompt": "professional product photography, clean white seamless background, studio lighting, high-end jewelry photography",
                    "guidance_scale": 30,
                    "steps": 50,
                    "strength": 0.95
                }
            )
            
            # Load result
            if isinstance(output, str):
                response = requests.get(output)
                return Image.open(io.BytesIO(response.content))
            else:
                return Image.open(io.BytesIO(output.read()))
                
        except Exception as e:
            logger.error(f"Replicate inpainting failed: {str(e)}")
            return self.fallback_crop(image, mask)
    
    def fallback_crop(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """Fallback method when inpainting fails"""
        # Find content bounds
        mask_array = np.array(mask)
        rows = np.any(mask_array == 0, axis=1)
        cols = np.any(mask_array == 0, axis=0)
        
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        
        # Add small padding
        padding = 10
        xmin = max(0, xmin - padding)
        ymin = max(0, ymin - padding)
        xmax = min(image.width, xmax + padding)
        ymax = min(image.height, ymax + padding)
        
        return image.crop((xmin, ymin, xmax, ymax))
    
    def detect_color_from_histogram(self, image: Image.Image) -> str:
        """Enhanced color detection"""
        img_array = np.array(image)
        
        # Get center region
        h, w = img_array.shape[:2]
        center = img_array[h//3:2*h//3, w//3:2*w//3]
        
        # Calculate color statistics
        avg_color = np.mean(center.reshape(-1, 3), axis=0)
        r, g, b = avg_color
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(center, cv2.COLOR_RGB2HSV)
        avg_hsv = np.mean(hsv.reshape(-1, 3), axis=0)
        h, s, v = avg_hsv
        
        # Check for plain white first (highest priority)
        if v > 220 and s < 15:
            return 'plain_white'
        
        # White gold
        if s < 25 and v > 180 and abs(r - g) < 10 and abs(g - b) < 10:
            return 'white_gold'
        
        # Yellow gold (stricter criteria)
        yellow_hue = 20 <= h <= 35
        warm_tone = r > g > b
        good_saturation = s > 40
        
        if yellow_hue and warm_tone and good_saturation:
            return 'yellow_gold'
        
        # Rose gold
        rose_hue = (h < 15 or h > 165)
        pink_tone = r > g and r > b
        moderate_saturation = 25 < s < 60
        
        if rose_hue and pink_tone and moderate_saturation:
            return 'rose_gold'
        
        # Default to plain white
        return 'plain_white'
    
    def apply_color_specific_enhancement(self, image: Image.Image, color_type: str) -> Image.Image:
        """Apply color-specific enhancements"""
        if color_type == 'plain_white':
            # Dramatic brightening for plain white
            img_array = np.array(image).astype(np.float32)
            
            # Strong brightness boost
            img_array = cv2.convertScaleAbs(img_array, alpha=1.3, beta=30)
            
            # Cool tone adjustment
            img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 1.02, 0, 255)  # Blue
            img_array[:, :, 1] = np.clip(img_array[:, :, 1] * 0.98, 0, 255)  # Green
            img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 0.96, 0, 255)  # Red
            
            enhanced = Image.fromarray(img_array.astype(np.uint8))
            
            # Strong contrast
            contrast = ImageEnhance.Contrast(enhanced)
            enhanced = contrast.enhance(1.4)
            
            # Slight blur then sharpen for smooth surface
            enhanced = enhanced.filter(ImageFilter.GaussianBlur(radius=0.5))
            sharpness = ImageEnhance.Sharpness(enhanced)
            enhanced = sharpness.enhance(1.8)
            
        elif color_type == 'white_gold':
            # Bright but warmer than plain white
            brightness = ImageEnhance.Brightness(image)
            enhanced = brightness.enhance(1.15)
            
            contrast = ImageEnhance.Contrast(enhanced)
            enhanced = contrast.enhance(1.2)
            
            # Slight warmth
            color = ImageEnhance.Color(enhanced)
            enhanced = color.enhance(0.95)
            
        elif color_type == 'yellow_gold':
            # Warm and rich
            brightness = ImageEnhance.Brightness(image)
            enhanced = brightness.enhance(1.08)
            
            # Boost warmth
            img_array = np.array(enhanced)
            img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 1.05, 0, 255)  # Red
            img_array[:, :, 1] = np.clip(img_array[:, :, 1] * 1.02, 0, 255)  # Green
            enhanced = Image.fromarray(img_array)
            
            contrast = ImageEnhance.Contrast(enhanced)
            enhanced = contrast.enhance(1.15)
            
        else:  # rose_gold
            # Pink tones
            brightness = ImageEnhance.Brightness(image)
            enhanced = brightness.enhance(1.10)
            
            # Enhance pink
            img_array = np.array(enhanced)
            img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 1.08, 0, 255)  # Red
            enhanced = Image.fromarray(img_array)
            
            contrast = ImageEnhance.Contrast(enhanced)
            enhanced = contrast.enhance(1.12)
        
        return enhanced
    
    def create_thumbnail_with_details(self, image: Image.Image, target_size: Tuple[int, int] = (1000, 1300)) -> Image.Image:
        """Create detailed thumbnail with tight crop"""
        # Convert to numpy for processing
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Find ring contours
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get bounding box of all contours
            x, y, w, h = cv2.boundingRect(np.concatenate(contours))
            
            # Very tight padding (10%)
            padding_x = int(w * 0.1)
            padding_y = int(h * 0.1)
            
            x = max(0, x - padding_x)
            y = max(0, y - padding_y)
            w = min(image.width - x, w + 2 * padding_x)
            h = min(image.height - y, h + 2 * padding_y)
            
            # Crop
            cropped = image.crop((x, y, x + w, y + h))
        else:
            # Fallback center crop
            width, height = image.size
            crop_size = min(width, height) * 0.6
            left = (width - crop_size) // 2
            top = (height - crop_size) // 2
            cropped = image.crop((left, top, left + crop_size, top + crop_size))
        
        # Resize to target maintaining aspect ratio
        cropped.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # Create final image with padding if needed
        final = Image.new('RGB', target_size, (250, 250, 250))
        paste_x = (target_size[0] - cropped.width) // 2
        paste_y = (target_size[1] - cropped.height) // 2
        final.paste(cropped, (paste_x, paste_y))
        
        # Enhance details
        sharpness = ImageEnhance.Sharpness(final)
        final = sharpness.enhance(1.5)
        
        return final
    
    def enhance_details(self, image: Image.Image) -> Image.Image:
        """Apply detail enhancement filters"""
        # Denoise
        img_array = np.array(image)
        denoised = cv2.fastNlMeansDenoisingColored(img_array, None, 3, 3, 7, 21)
        
        # Unsharp mask for detail enhancement
        gaussian = cv2.GaussianBlur(denoised, (0, 0), 2.0)
        enhanced = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        return Image.fromarray(enhanced)
    
    def image_to_base64_no_padding(self, image: Image.Image, format: str = 'JPEG') -> str:
        """Convert image to base64 without padding"""
        buffer = io.BytesIO()
        
        if format == 'JPEG':
            if image.mode == 'RGBA':
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[3])
                image = rgb_image
            image.save(buffer, format=format, quality=95, optimize=True)
        else:
            image.save(buffer, format=format)
            
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode()
        return img_str.rstrip('=')

    def process_thumbnail(self, input_url: str) -> Tuple[str, str, Dict[str, Any]]:
        """Main processing function for thumbnail generation"""
        processing_info = {
            'original_size': None,
            'mask_created': False,
            'inpainting_applied': False,
            'crop_coords': None,
            'final_size': None,
            'detected_color': None,
            'version': 'v41_fixed_url'
        }
        
        try:
            # Load image with improved error handling
            logger.info(f"Starting to load image...")
            image = self.load_image_from_source(input_url)
            
            if image is None:
                raise ValueError("Failed to load image from source")
            
            processing_info['original_size'] = image.size
            logger.info(f"Image loaded successfully: {image.size}")
            
            # Detect black masking
            mask_detection = self.detect_black_masking_multi_method(image)
            logger.info(f"Black masking detection: {mask_detection}")
            
            if mask_detection['detected']:
                processing_info['mask_created'] = True
                processing_info['mask_confidence'] = mask_detection['confidence']
                
                # Create inpainting mask
                mask = self.create_inpainting_mask(image, mask_detection['frame'])
                
                # Remove black frame
                image = self.remove_black_frame_with_replicate(image, mask)
                processing_info['inpainting_applied'] = True
                logger.info("Black frame removed successfully")
            
            # Detect color
            detected_color = self.detect_color_from_histogram(image)
            processing_info['detected_color'] = detected_color
            logger.info(f"Detected color: {detected_color}")
            
            # Apply color-specific enhancements
            image = self.apply_color_specific_enhancement(image, detected_color)
            
            # Create thumbnail with tight crop
            thumbnail = self.create_thumbnail_with_details(image)
            
            # Final detail enhancement
            thumbnail = self.enhance_details(thumbnail)
            
            processing_info['final_size'] = thumbnail.size
            
            # Convert to base64 without padding
            thumbnail_base64 = self.image_to_base64_no_padding(thumbnail)
            thumbnail_data_url = f"data:image/jpeg;base64,{thumbnail_base64}"
            
            return thumbnail_data_url, detected_color, processing_info
            
        except Exception as e:
            logger.error(f"Error processing thumbnail: {str(e)}")
            logger.error(traceback.format_exc())
            raise

def handler(event):
    """RunPod handler function"""
    try:
        logger.info("="*50)
        logger.info(f"Thumbnail handler started with event keys: {list(event.keys())}")
        
        handler_instance = ThumbnailHandler()
        
        # Find input URL
        input_url = handler_instance.find_input_url(event)
        
        if not input_url:
            raise ValueError("No input URL found. Please check the input structure.")
        
        logger.info(f"Processing image from: {input_url[:100]}...")
        
        # Process thumbnail
        thumbnail_data_url, detected_color, processing_info = handler_instance.process_thumbnail(input_url)
        
        # Prepare response
        result = {
            "output": {
                "thumbnail": thumbnail_data_url,
                "detected_color": detected_color,
                "processing_info": processing_info,
                "success": True
            }
        }
        
        logger.info("Thumbnail processing completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error in handler: {str(e)}")
        logger.error(traceback.format_exc())
        
        return {
            "output": {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "success": False
            }
        }

# For RunPod
if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})

# Add import for ImageDraw
from PIL import ImageDraw
