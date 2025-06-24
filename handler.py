import runpod
import requests
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
import base64
import traceback
from typing import Dict, Any, Tuple, Optional
import replicate
import os
import time
import cv2
from scipy import ndimage
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThumbnailHandler:
    def __init__(self):
        """Initialize thumbnail handler with Replicate client"""
        logger.info("Thumbnail Handler v40 initialized - Stable Version")
        self.replicate_client = replicate.Client(api_token=os.environ.get("REPLICATE_API_TOKEN"))
        self.ring_colors = {
            'yellow_gold': {
                'ranges': [
                    ((15, 30, 100), (35, 255, 255)),
                    ((20, 40, 120), (40, 255, 255))
                ],
                'display': 'Yellow Gold'
            },
            'rose_gold': {
                'ranges': [
                    ((0, 20, 100), (15, 255, 255)),
                    ((340, 20, 100), (359, 255, 255))
                ],
                'display': 'Rose Gold'
            },
            'white_gold': {
                'ranges': [
                    ((0, 0, 180), (180, 10, 255)),
                    ((0, 0, 200), (180, 5, 255))
                ],
                'display': 'White Gold'
            },
            'unplated_white': {
                'ranges': [
                    ((0, 0, 120), (180, 15, 180))
                ],
                'display': 'Unplated White'
            }
        }

    def find_input_url(self, event: Dict[str, Any]) -> Optional[str]:
        """Find input URL from various possible paths in the event"""
        logger.info("Searching for input URL in event structure...")
        
        # Direct input paths
        input_data = event.get('input', {})
        
        # Try direct URL keys
        url_keys = ['input_url', 'image_url', 'url', 'image', 'enhanced_image']
        for key in url_keys:
            if key in input_data and input_data[key]:
                logger.info(f"Found URL in direct input: {key}")
                return input_data[key]
        
        # Try Make.com nested structures
        # Pattern: data.output.output
        if 'data' in input_data and isinstance(input_data['data'], dict):
            data = input_data['data']
            if 'output' in data and isinstance(data['output'], dict):
                output = data['output']
                if 'output' in output and isinstance(output['output'], dict):
                    nested_output = output['output']
                    for key in url_keys:
                        if key in nested_output and nested_output[key]:
                            logger.info(f"Found URL in data.output.output: {key}")
                            return nested_output[key]
                else:
                    for key in url_keys:
                        if key in output and output[key]:
                            logger.info(f"Found URL in data.output: {key}")
                            return output[key]
        
        # Try numbered keys (like '4')
        for num_key in input_data:
            if num_key.isdigit() and isinstance(input_data[num_key], dict):
                numbered_data = input_data[num_key]
                if 'data' in numbered_data and isinstance(numbered_data['data'], dict):
                    data = numbered_data['data']
                    if 'output' in data and isinstance(data['output'], dict):
                        output = data['output']
                        if 'output' in output and isinstance(output['output'], dict):
                            nested_output = output['output']
                            for key in url_keys:
                                if key in nested_output and nested_output[key]:
                                    logger.info(f"Found URL in {num_key}.data.output.output: {key}")
                                    return nested_output[key]
        
        # Log what we found for debugging
        logger.warning(f"Could not find URL. Event keys: {list(event.keys())}")
        if input_data:
            logger.warning(f"Input keys: {list(input_data.keys())}")
        
        return None

    def load_image_from_source(self, source: str) -> Optional[Image.Image]:
        """Load image from URL or base64 string with improved error handling"""
        try:
            # URL case
            if source.startswith(('http://', 'https://')):
                logger.info(f"Loading image from URL: {source[:100]}...")
                
                # Multiple retry attempts with different headers
                headers_list = [
                    {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'},
                    {'User-Agent': 'Python-Requests/2.31.0'},
                    {}  # No headers
                ]
                
                for idx, headers in enumerate(headers_list):
                    try:
                        logger.info(f"Attempt {idx + 1} with headers: {headers}")
                        response = requests.get(source, headers=headers, timeout=30, stream=True)
                        if response.status_code == 200:
                            image_data = io.BytesIO(response.content)
                            img = Image.open(image_data)
                            
                            # Convert to RGB if necessary
                            if img.mode not in ('RGB', 'RGBA'):
                                img = img.convert('RGB')
                            
                            logger.info(f"Successfully loaded image: {img.size}, mode: {img.mode}")
                            return img
                        else:
                            logger.warning(f"HTTP {response.status_code} for attempt {idx + 1}")
                    except Exception as e:
                        logger.warning(f"Attempt {idx + 1} failed: {str(e)}")
                        continue
                
                logger.error(f"All URL loading attempts failed")
                return None
                
            # Base64 case
            elif source.startswith('data:image'):
                logger.info("Loading image from base64 data URL")
                base64_str = source.split(',')[1]
                
                # Fix padding if needed
                padding = 4 - len(base64_str) % 4
                if padding != 4:
                    base64_str += '=' * padding
                    
                image_data = base64.b64decode(base64_str)
                img = Image.open(io.BytesIO(image_data))
                if img.mode not in ('RGB', 'RGBA'):
                    img = img.convert('RGB')
                return img
                
            # Raw base64 case
            else:
                logger.info("Loading image from raw base64")
                # Handle potential padding issues
                padding = 4 - len(source) % 4
                if padding != 4:
                    source += '=' * padding
                    
                image_data = base64.b64decode(source)
                img = Image.open(io.BytesIO(image_data))
                if img.mode not in ('RGB', 'RGBA'):
                    img = img.convert('RGB')
                return img
                
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            traceback.print_exc()
            return None

    def create_multi_stage_mask(self, image: Image.Image) -> np.ndarray:
        """Create detailed multi-stage mask for black area detection"""
        img_array = np.array(image)
        
        # Convert to multiple color spaces for comprehensive detection
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Stage 1: HSV-based detection with multiple thresholds
        black_masks_hsv = []
        
        # Very dark areas
        black_masks_hsv.append(cv2.inRange(hsv, (0, 0, 0), (180, 255, 30)))
        
        # Dark areas with some saturation
        black_masks_hsv.append(cv2.inRange(hsv, (0, 0, 0), (180, 50, 50)))
        
        # Low value areas
        black_masks_hsv.append(cv2.inRange(hsv, (0, 0, 0), (180, 30, 70)))
        
        # Stage 2: LAB-based detection
        black_masks_lab = []
        
        # Low lightness
        black_masks_lab.append(cv2.inRange(lab[:,:,0], 0, 30))
        black_masks_lab.append(cv2.inRange(lab[:,:,0], 0, 40))
        
        # Stage 3: Grayscale detection
        black_masks_gray = []
        black_masks_gray.append(cv2.inRange(gray, 0, 30))
        black_masks_gray.append(cv2.inRange(gray, 0, 45))
        
        # Stage 4: Edge-based detection for boundaries
        edges = cv2.Canny(gray, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Combine all masks with weights
        combined_mask = np.zeros_like(black_masks_hsv[0], dtype=np.float32)
        
        # HSV masks (highest weight)
        for mask in black_masks_hsv:
            combined_mask += mask.astype(np.float32) * 0.3
            
        # LAB masks
        for mask in black_masks_lab:
            combined_mask += mask.astype(np.float32) * 0.25
            
        # Gray masks
        for mask in black_masks_gray:
            combined_mask += mask.astype(np.float32) * 0.2
            
        # Edge information
        combined_mask += edges_dilated.astype(np.float32) * 0.1
        
        # Normalize and threshold
        combined_mask = np.clip(combined_mask, 0, 255).astype(np.uint8)
        _, final_mask = cv2.threshold(combined_mask, 50, 255, cv2.THRESH_BINARY)
        
        # Multi-stage morphological operations
        # Stage 1: Remove noise
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_small)
        
        # Stage 2: Fill small gaps
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_medium)
        
        # Stage 3: Smooth boundaries
        final_mask = cv2.dilate(final_mask, kernel_medium, iterations=2)
        final_mask = cv2.erode(final_mask, kernel_medium, iterations=2)
        
        # Stage 4: Final refinement
        final_mask = cv2.medianBlur(final_mask, 5)
        
        return final_mask

    def remove_masking_with_replicate(self, image: Image.Image, mask: np.ndarray, max_retries: int = 3) -> Image.Image:
        """Remove black masking using Replicate API with retry logic"""
        try:
            # Convert mask to PIL Image
            mask_image = Image.fromarray(mask)
            
            # Prepare images for Replicate
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            mask_buffer = io.BytesIO()
            mask_image.save(mask_buffer, format='PNG')
            mask_buffer.seek(0)
            
            # Try different Replicate models
            models = [
                "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3",
                "andreasjansson/stable-diffusion-inpainting:e490d072a34a94a11e9711ed5a6ba621c3fab884eda1665d9d3a282d65a21180"
            ]
            
            for model in models:
                for attempt in range(max_retries):
                    try:
                        logger.info(f"Attempting inpainting with model {model.split('/')[1]}, attempt {attempt + 1}")
                        
                        output = self.replicate_client.run(
                            model,
                            input={
                                "image": img_buffer,
                                "mask": mask_buffer,
                                "prompt": "high quality wedding ring, professional jewelry photography, clean white background, detailed metal surface",
                                "negative_prompt": "black areas, shadows, dark spots, masking tape, blur, low quality",
                                "num_inference_steps": 50,
                                "guidance_scale": 7.5
                            }
                        )
                        
                        if output and len(output) > 0:
                            inpainted_url = output[0] if isinstance(output, list) else output
                            response = requests.get(inpainted_url)
                            if response.status_code == 200:
                                return Image.open(io.BytesIO(response.content))
                                
                    except Exception as e:
                        logger.warning(f"Inpainting attempt {attempt + 1} failed: {str(e)}")
                        if attempt < max_retries - 1:
                            time.sleep(2)
                        continue
            
            # If all attempts fail, apply fallback enhancement
            logger.warning("All inpainting attempts failed, using fallback enhancement")
            return self.apply_fallback_enhancement(image, mask)
            
        except Exception as e:
            logger.error(f"Error in remove_masking_with_replicate: {str(e)}")
            return self.apply_fallback_enhancement(image, mask)

    def apply_fallback_enhancement(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        """Apply fallback enhancement when inpainting fails"""
        img_array = np.array(image)
        
        # Invert mask to get areas to keep
        keep_mask = cv2.bitwise_not(mask)
        
        # Create a bright background
        bright_bg = np.full_like(img_array, 245)
        
        # Apply mask
        result = np.where(keep_mask[..., np.newaxis], img_array, bright_bg)
        
        # Apply slight blur to blend edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_blurred = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0) / 255.0
        
        result = (img_array * (1 - mask_blurred[..., np.newaxis]) + 
                 bright_bg * mask_blurred[..., np.newaxis]).astype(np.uint8)
        
        return Image.fromarray(result)

    def detect_ring_color(self, image: Image.Image) -> str:
        """Detect ring color from the image"""
        try:
            img_array = np.array(image)
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            
            # Create mask for non-background areas
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            _, ring_mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
            
            # Erode mask to focus on ring center
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            ring_mask = cv2.erode(ring_mask, kernel, iterations=2)
            
            best_color = 'white_gold'
            max_score = 0
            
            for color_name, color_info in self.ring_colors.items():
                total_pixels = 0
                
                for hsv_range in color_info['ranges']:
                    mask = cv2.inRange(hsv, hsv_range[0], hsv_range[1])
                    mask = cv2.bitwise_and(mask, ring_mask)
                    total_pixels += cv2.countNonZero(mask)
                
                if total_pixels > max_score:
                    max_score = total_pixels
                    best_color = color_name
            
            # Additional brightness check for white metals
            if best_color in ['white_gold', 'unplated_white']:
                bright_pixels = cv2.inRange(img_array, (180, 180, 180), (255, 255, 255))
                bright_pixels = cv2.bitwise_and(bright_pixels, ring_mask)
                bright_count = cv2.countNonZero(bright_pixels)
                
                if bright_count > cv2.countNonZero(ring_mask) * 0.3:
                    avg_brightness = np.mean(img_array[ring_mask > 0])
                    best_color = 'white_gold' if avg_brightness > 200 else 'unplated_white'
            
            return self.ring_colors[best_color]['display']
            
        except Exception as e:
            logger.error(f"Error in color detection: {str(e)}")
            return 'White Gold'

    def find_optimal_crop(self, image: Image.Image, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Find optimal crop area focusing on the ring"""
        h, w = mask.shape
        
        # Invert mask to get ring area
        ring_mask = cv2.bitwise_not(mask)
        
        # Find contours
        contours, _ = cv2.findContours(ring_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # Fallback to center crop
            return self.calculate_center_crop(w, h)
        
        # Find largest contour (main ring)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(largest_contour)
        
        # Calculate center of ring
        cx = x + cw // 2
        cy = y + ch // 2
        
        # Calculate crop dimensions (1000x1300)
        crop_w, crop_h = 1000, 1300
        
        # Calculate crop boundaries with ring centered
        left = max(0, cx - crop_w // 2)
        top = max(0, cy - crop_h // 2)
        right = min(w, left + crop_w)
        bottom = min(h, top + crop_h)
        
        # Adjust if crop goes out of bounds
        if right - left < crop_w:
            if left == 0:
                right = min(w, crop_w)
            else:
                left = max(0, right - crop_w)
        
        if bottom - top < crop_h:
            if top == 0:
                bottom = min(h, crop_h)
            else:
                top = max(0, bottom - crop_h)
        
        return left, top, right, bottom

    def calculate_center_crop(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """Calculate center crop coordinates"""
        crop_w, crop_h = 1000, 1300
        left = max(0, (width - crop_w) // 2)
        top = max(0, (height - crop_h) // 2)
        right = min(width, left + crop_w)
        bottom = min(height, top + crop_h)
        return left, top, right, bottom

    def enhance_ring_details(self, image: Image.Image) -> Image.Image:
        """Enhance ring details with advanced processing"""
        # Apply unsharp mask for detail enhancement
        gaussian = image.filter(ImageFilter.GaussianBlur(radius=2))
        img_array = np.array(image).astype(np.float32)
        gaussian_array = np.array(gaussian).astype(np.float32)
        
        # Unsharp mask
        enhanced = img_array + 1.5 * (img_array - gaussian_array)
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        enhanced_img = Image.fromarray(enhanced)
        
        # Adjust brightness and contrast
        brightness_enhancer = ImageEnhance.Brightness(enhanced_img)
        enhanced_img = brightness_enhancer.enhance(1.1)
        
        contrast_enhancer = ImageEnhance.Contrast(enhanced_img)
        enhanced_img = contrast_enhancer.enhance(1.2)
        
        # Slight sharpening
        enhanced_img = enhanced_img.filter(ImageFilter.SHARPEN)
        
        return enhanced_img

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
            'version': 'v40_stable'
        }
        
        try:
            # Load image with improved error handling
            logger.info(f"Starting to load image from: {input_url[:100]}...")
            image = self.load_image_from_source(input_url)
            
            if image is None:
                raise ValueError("Failed to load image from source")
            
            processing_info['original_size'] = image.size
            logger.info(f"Image loaded successfully: {image.size}")
            
            # Ensure image is in correct format
            if image.size != (6720, 4480):
                logger.info(f"Resizing image from {image.size} to (6720, 4480)")
                image = image.resize((6720, 4480), Image.Resampling.LANCZOS)
            
            # Create multi-stage mask
            logger.info("Creating multi-stage mask...")
            mask = self.create_multi_stage_mask(image)
            processing_info['mask_created'] = True
            
            # Remove masking
            logger.info("Removing black masking...")
            cleaned_image = self.remove_masking_with_replicate(image, mask)
            processing_info['inpainting_applied'] = True
            
            # Find optimal crop
            logger.info("Finding optimal crop area...")
            crop_coords = self.find_optimal_crop(cleaned_image, mask)
            processing_info['crop_coords'] = crop_coords
            
            # Crop image
            cropped = cleaned_image.crop(crop_coords)
            
            # Resize to exact thumbnail size
            thumbnail = cropped.resize((1000, 1300), Image.Resampling.LANCZOS)
            processing_info['final_size'] = thumbnail.size
            
            # Detect color
            logger.info("Detecting ring color...")
            detected_color = self.detect_ring_color(thumbnail)
            processing_info['detected_color'] = detected_color
            
            # Enhance details
            logger.info("Enhancing ring details...")
            final_image = self.enhance_ring_details(thumbnail)
            
            # Convert to base64
            logger.info("Converting to base64...")
            base64_image = self.image_to_base64_no_padding(final_image)
            
            return base64_image, detected_color, processing_info
            
        except Exception as e:
            logger.error(f"Error in process_thumbnail: {str(e)}")
            traceback.print_exc()
            raise

def handler(event):
    """RunPod handler function with proper return structure"""
    logger.info("=== Thumbnail Handler v40 Started ===")
    
    try:
        handler_instance = ThumbnailHandler()
        
        # Find input URL from various possible paths
        input_url = handler_instance.find_input_url(event)
        
        if not input_url:
            raise ValueError("No input URL found in event. Please check the input structure.")
        
        logger.info(f"Processing image from URL: {input_url[:100]}...")
        
        # Process thumbnail
        processed_image, detected_color, processing_info = handler_instance.process_thumbnail(input_url)
        
        # Return structure matching Make.com expectations
        result = {
            "output": {
                "processed_image": f"data:image/jpeg;base64,{processed_image}",
                "detected_color": detected_color,
                "processing_info": processing_info,
                "success": True
            }
        }
        
        logger.info("Processing completed successfully")
        logger.info(f"Detected color: {detected_color}")
        logger.info(f"Final size: {processing_info['final_size']}")
        
        return result
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error in handler: {error_msg}")
        traceback.print_exc()
        
        return {
            "output": {
                "error": error_msg,
                "traceback": traceback.format_exc(),
                "success": False
            }
        }

# RunPod endpoint
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
