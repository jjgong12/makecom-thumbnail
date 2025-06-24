import os
import json
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import logging
import traceback
import base64
from io import BytesIO
import requests
from typing import Dict, Any, Tuple, List, Optional
import time
import replicate
from datetime import datetime
import colorsys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThumbnailHandler:
    def __init__(self):
        """Initialize the Thumbnail Handler"""
        self.setup_environment()
        
    def setup_environment(self):
        """Set up environment and API keys"""
        replicate_api_token = os.environ.get('REPLICATE_API_TOKEN')
        if replicate_api_token:
            os.environ['REPLICATE_API_TOKEN'] = replicate_api_token
            self.replicate_available = True
            logger.info("Replicate API token found and set")
        else:
            self.replicate_available = False
            logger.warning("Replicate API token not found")

    def detect_black_frame_enhanced(self, image: np.ndarray) -> Dict[str, Any]:
        """Enhanced black frame detection with multiple validation steps"""
        height, width = image.shape[:2]
        edge_thickness = min(width, height) // 10  # 10% of smaller dimension
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Multi-step detection
        detection_results = {
            'edge_analysis': self._analyze_edges(gray, edge_thickness),
            'corner_analysis': self._analyze_corners(gray, edge_thickness),
            'gradient_analysis': self._analyze_gradients(gray),
            'connectivity_analysis': self._analyze_connectivity(gray, edge_thickness)
        }
        
        # Weighted scoring
        score = (
            detection_results['edge_analysis']['score'] * 0.3 +
            detection_results['corner_analysis']['score'] * 0.3 +
            detection_results['gradient_analysis']['score'] * 0.2 +
            detection_results['connectivity_analysis']['score'] * 0.2
        )
        
        has_frame = score > 0.6
        
        # Find frame boundaries if detected
        if has_frame:
            boundaries = self._find_frame_boundaries(gray, edge_thickness)
        else:
            boundaries = None
            
        logger.info(f"Black frame detection - Score: {score:.2f}, Has frame: {has_frame}")
        
        return {
            'has_black_frame': has_frame,
            'confidence_score': float(score),
            'detection_details': detection_results,
            'frame_boundaries': boundaries
        }
    
    def _analyze_edges(self, gray: np.ndarray, thickness: int) -> Dict[str, Any]:
        """Analyze edges for black pixels"""
        height, width = gray.shape
        
        # Sample edges with some margin
        edges = {
            'top': gray[0:thickness, :],
            'bottom': gray[height-thickness:, :],
            'left': gray[:, 0:thickness],
            'right': gray[:, width-thickness:]
        }
        
        scores = {}
        for edge_name, edge_region in edges.items():
            # Multiple threshold checks
            black_pixels_strict = np.sum(edge_region < 30)
            black_pixels_medium = np.sum(edge_region < 50)
            black_pixels_loose = np.sum(edge_region < 70)
            
            total_pixels = edge_region.size
            
            # Weighted score
            score = (
                (black_pixels_strict / total_pixels) * 0.5 +
                (black_pixels_medium / total_pixels) * 0.3 +
                (black_pixels_loose / total_pixels) * 0.2
            )
            scores[edge_name] = score
        
        avg_score = np.mean(list(scores.values()))
        
        return {
            'score': avg_score,
            'edge_scores': scores
        }
    
    def _analyze_corners(self, gray: np.ndarray, thickness: int) -> Dict[str, Any]:
        """Analyze corners for black frame continuity"""
        height, width = gray.shape
        corner_size = thickness * 2
        
        corners = {
            'top_left': gray[0:corner_size, 0:corner_size],
            'top_right': gray[0:corner_size, width-corner_size:],
            'bottom_left': gray[height-corner_size:, 0:corner_size],
            'bottom_right': gray[height-corner_size:, width-corner_size:]
        }
        
        scores = {}
        for corner_name, corner_region in corners.items():
            black_ratio = np.sum(corner_region < 50) / corner_region.size
            scores[corner_name] = black_ratio
        
        avg_score = np.mean(list(scores.values()))
        
        return {
            'score': avg_score,
            'corner_scores': scores
        }
    
    def _analyze_gradients(self, gray: np.ndarray) -> Dict[str, Any]:
        """Analyze gradients to detect sharp transitions"""
        # Sobel edge detection
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Look for strong gradients near edges
        height, width = gray.shape
        edge_width = min(width, height) // 8
        
        edge_gradients = []
        
        # Top edge
        edge_gradients.append(gradient_magnitude[edge_width:edge_width*2, :].mean())
        # Bottom edge
        edge_gradients.append(gradient_magnitude[height-edge_width*2:height-edge_width, :].mean())
        # Left edge
        edge_gradients.append(gradient_magnitude[:, edge_width:edge_width*2].mean())
        # Right edge
        edge_gradients.append(gradient_magnitude[:, width-edge_width*2:width-edge_width].mean())
        
        avg_gradient = np.mean(edge_gradients)
        score = min(avg_gradient / 100, 1.0)  # Normalize
        
        return {
            'score': score,
            'average_gradient': avg_gradient
        }
    
    def _analyze_connectivity(self, gray: np.ndarray, thickness: int) -> Dict[str, Any]:
        """Check if black regions are connected (forming a frame)"""
        # Binary threshold
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        
        # Invert so black becomes white
        binary_inv = 255 - binary
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(binary_inv)
        
        # Check if there's a large connected component near edges
        height, width = gray.shape
        edge_mask = np.zeros_like(labels, dtype=bool)
        edge_mask[0:thickness, :] = True
        edge_mask[height-thickness:, :] = True
        edge_mask[:, 0:thickness] = True
        edge_mask[:, width-thickness:] = True
        
        # Count pixels in edge regions for each component
        edge_component_sizes = []
        for label in range(1, num_labels):  # Skip background
            component_mask = labels == label
            edge_pixels = np.sum(component_mask & edge_mask)
            if edge_pixels > 0:
                total_pixels = np.sum(component_mask)
                edge_component_sizes.append(total_pixels)
        
        if edge_component_sizes:
            largest_component = max(edge_component_sizes)
            total_edge_area = np.sum(edge_mask)
            score = min(largest_component / total_edge_area, 1.0)
        else:
            score = 0.0
        
        return {
            'score': score,
            'num_edge_components': len(edge_component_sizes)
        }
    
    def _find_frame_boundaries(self, gray: np.ndarray, initial_thickness: int) -> Optional[Dict[str, int]]:
        """Find exact boundaries of the black frame"""
        height, width = gray.shape
        threshold = 70  # Slightly loose threshold
        
        # Find boundaries from each edge
        boundaries = {
            'top': 0,
            'bottom': height,
            'left': 0,
            'right': width
        }
        
        # Scan from top
        for y in range(min(height//3, initial_thickness*2)):
            if np.mean(gray[y, :]) > threshold:
                boundaries['top'] = y
                break
        
        # Scan from bottom
        for y in range(height-1, max(height*2//3, height-initial_thickness*2), -1):
            if np.mean(gray[y, :]) > threshold:
                boundaries['bottom'] = y + 1
                break
        
        # Scan from left
        for x in range(min(width//3, initial_thickness*2)):
            if np.mean(gray[:, x]) > threshold:
                boundaries['left'] = x
                break
        
        # Scan from right
        for x in range(width-1, max(width*2//3, width-initial_thickness*2), -1):
            if np.mean(gray[:, x]) > threshold:
                boundaries['right'] = x + 1
                break
        
        return boundaries

    def remove_black_masking_with_replicate(self, image: np.ndarray, boundaries: Dict[str, int]) -> np.ndarray:
        """Remove black masking using Replicate API with better prompting"""
        try:
            # Create a more aggressive mask
            height, width = image.shape[:2]
            
            # Expand boundaries slightly for better inpainting
            expand_pixels = 10
            mask_boundaries = {
                'top': max(0, boundaries['top'] - expand_pixels),
                'bottom': min(height, boundaries['bottom'] + expand_pixels),
                'left': max(0, boundaries['left'] - expand_pixels),
                'right': min(width, boundaries['right'] + expand_pixels)
            }
            
            # Create mask
            mask = np.ones((height, width), dtype=np.uint8) * 255
            
            # Mark frame areas as black (to be inpainted)
            mask[mask_boundaries['top']:mask_boundaries['top']+boundaries['top']+expand_pixels, :] = 0
            mask[mask_boundaries['bottom']-boundaries['bottom']-expand_pixels:mask_boundaries['bottom'], :] = 0
            mask[:, mask_boundaries['left']:mask_boundaries['left']+boundaries['left']+expand_pixels] = 0
            mask[:, mask_boundaries['right']-boundaries['right']-expand_pixels:mask_boundaries['right']] = 0
            
            # Convert images to base64
            img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            mask_pil = Image.fromarray(mask)
            
            # Convert to base64
            img_buffer = BytesIO()
            mask_buffer = BytesIO()
            
            img_pil.save(img_buffer, format='PNG')
            mask_pil.save(mask_buffer, format='PNG')
            
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode('utf-8')
            
            # Use Replicate API
            output = replicate.run(
                "stability-ai/stable-diffusion-inpainting",
                input={
                    "image": f"data:image/png;base64,{img_base64}",
                    "mask": f"data:image/png;base64,{mask_base64}",
                    "prompt": "clean white background, product photography background, seamless studio background",
                    "negative_prompt": "black frame, black border, dark edges, shadows",
                    "num_inference_steps": 30,
                    "guidance_scale": 7.5,
                }
            )
            
            # Process result
            if output and len(output) > 0:
                result_url = output[0] if isinstance(output, list) else output
                response = requests.get(result_url)
                result_image = Image.open(BytesIO(response.content))
                result_array = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
                
                logger.info("Successfully removed black masking with Replicate")
                return result_array
            else:
                logger.warning("No output from Replicate, returning original")
                return image
                
        except Exception as e:
            logger.error(f"Error in Replicate inpainting: {str(e)}")
            return image

    def detect_ring_color_in_thumbnail(self, image: np.ndarray) -> str:
        """Detect ring color with improved logic for white gold"""
        # Convert to RGB for better color analysis
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = rgb_image.shape[:2]
        
        # Focus on center area where ring is likely to be
        center_y, center_x = height // 2, width // 2
        roi_size = min(width, height) // 3
        
        roi = rgb_image[
            max(0, center_y - roi_size):min(height, center_y + roi_size),
            max(0, center_x - roi_size):min(width, center_x + roi_size)
        ]
        
        # Convert to HSV for better color detection
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        
        # Get average values
        avg_hue = np.mean(hsv_roi[:, :, 0])
        avg_sat = np.mean(hsv_roi[:, :, 1])
        avg_val = np.mean(hsv_roi[:, :, 2])
        
        # Also check RGB values
        avg_r = np.mean(roi[:, :, 0])
        avg_g = np.mean(roi[:, :, 1])
        avg_b = np.mean(roi[:, :, 2])
        
        # Color detection logic with stricter yellow detection
        color = "white_gold"  # Default to white gold
        
        # Yellow gold - must be clearly yellow
        if (15 <= avg_hue <= 35 and avg_sat > 50 and avg_val > 100 and
            avg_r > avg_b + 30 and avg_g > avg_b + 20):
            color = "yellow_gold"
        
        # Rose gold - pinkish hue
        elif ((0 <= avg_hue <= 15 or 340 <= avg_hue <= 360) and 
              avg_sat > 30 and avg_r > avg_g + 10 and avg_r > avg_b + 20):
            color = "rose_gold"
        
        # Unplated white - very low saturation, high brightness
        elif avg_sat < 20 and avg_val > 180 and abs(avg_r - avg_g) < 10 and abs(avg_g - avg_b) < 10:
            color = "unplated_white"
        
        logger.info(f"Thumbnail color detection - HSV: ({avg_hue:.1f}, {avg_sat:.1f}, {avg_val:.1f}), "
                   f"RGB: ({avg_r:.1f}, {avg_g:.1f}, {avg_b:.1f}) -> {color}")
        
        return color

    def apply_color_specific_enhancement(self, image: np.ndarray, color: str) -> np.ndarray:
        """Apply color-specific enhancements with focus on white gold clarity"""
        enhanced = image.copy()
        
        if color == "unplated_white":
            # Strong enhancement for pure white appearance
            # Increase brightness significantly
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.3, beta=30)
            
            # Cool down the tone (remove yellow cast)
            enhanced[:, :, 0] = np.clip(enhanced[:, :, 0] * 1.05, 0, 255)  # Boost blue
            enhanced[:, :, 1] = np.clip(enhanced[:, :, 1] * 0.98, 0, 255)  # Slight green reduction
            enhanced[:, :, 2] = np.clip(enhanced[:, :, 2] * 0.95, 0, 255)  # Reduce red
            
            # Increase contrast
            pil_img = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(1.4)
            
            # Remove noise for cleaner surface
            pil_img = pil_img.filter(ImageFilter.MedianFilter(size=3))
            
            enhanced = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
        elif color == "white_gold":
            # Moderate enhancement for white gold
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=20)
            
            # Slight cool tone
            enhanced[:, :, 0] = np.clip(enhanced[:, :, 0] * 1.02, 0, 255)
            enhanced[:, :, 2] = np.clip(enhanced[:, :, 2] * 0.98, 0, 255)
            
            # Moderate contrast boost
            pil_img = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(1.3)
            enhanced = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
        elif color == "yellow_gold":
            # Warm enhancement for yellow gold
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.15, beta=15)
            
            # Enhance yellow tones
            enhanced[:, :, 1] = np.clip(enhanced[:, :, 1] * 1.05, 0, 255)  # Green
            enhanced[:, :, 2] = np.clip(enhanced[:, :, 2] * 1.08, 0, 255)  # Red
            
        elif color == "rose_gold":
            # Pink/rose enhancement
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.15, beta=15)
            
            # Enhance pink tones
            enhanced[:, :, 2] = np.clip(enhanced[:, :, 2] * 1.1, 0, 255)  # Red
            enhanced[:, :, 0] = np.clip(enhanced[:, :, 0] * 0.95, 0, 255)  # Reduce blue
        
        return enhanced

    def enhance_details(self, image: np.ndarray, color: str) -> np.ndarray:
        """Enhanced detail processing with special handling for white metals"""
        # Denoise first
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        
        # Different sharpening for different colors
        if color in ["unplated_white", "white_gold"]:
            # Stronger sharpening for white metals
            kernel = np.array([[-1, -1, -1],
                               [-1, 9.5, -1],
                               [-1, -1, -1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)
            
            # Additional clarity enhancement
            pil_img = Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
            
            # Unsharp mask for fine details
            blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=2))
            sharpened_pil = Image.blend(blurred, pil_img, 1.5)
            
            result = cv2.cvtColor(np.array(sharpened_pil), cv2.COLOR_RGB2BGR)
            
        else:
            # Moderate sharpening for gold colors
            kernel = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
            result = cv2.filter2D(denoised, -1, kernel)
        
        return result

    def process_thumbnail(self, input_path: str) -> Tuple[np.ndarray, str, Dict[str, Any]]:
        """Main thumbnail processing function"""
        logger.info("Starting thumbnail processing...")
        
        # Load image
        if input_path.startswith('http'):
            response = requests.get(input_path)
            image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        else:
            image = cv2.imread(input_path)
        
        if image is None:
            raise ValueError("Failed to load image")
        
        original_shape = image.shape
        logger.info(f"Original image shape: {original_shape}")
        
        # Detect black frame with enhanced algorithm
        frame_detection = self.detect_black_frame_enhanced(image)
        
        # Remove black masking if detected
        if frame_detection['has_black_frame'] and self.replicate_available:
            logger.info("Black frame detected, removing with Replicate...")
            image = self.remove_black_masking_with_replicate(
                image, 
                frame_detection['frame_boundaries']
            )
        
        # Detect ring color
        detected_color = self.detect_ring_color_in_thumbnail(image)
        logger.info(f"Detected ring color: {detected_color}")
        
        # Crop to focus on ring
        height, width = image.shape[:2]
        crop_margin = 0.15
        
        y1 = int(height * crop_margin)
        y2 = int(height * (1 - crop_margin))
        x1 = int(width * crop_margin)
        x2 = int(width * (1 - crop_margin))
        
        cropped = image[y1:y2, x1:x2]
        
        # Apply color-specific enhancement
        enhanced = self.apply_color_specific_enhancement(cropped, detected_color)
        
        # Enhance details
        detailed = self.enhance_details(enhanced, detected_color)
        
        # Resize to target dimensions
        target_width, target_height = 1000, 1300
        resized = cv2.resize(detailed, (target_width, target_height), 
                           interpolation=cv2.INTER_LANCZOS4)
        
        # Final quality boost
        final_enhanced = cv2.convertScaleAbs(resized, alpha=1.05, beta=5)
        
        processing_info = {
            'original_shape': original_shape,
            'detected_color': detected_color,
            'has_black_frame': frame_detection['has_black_frame'],
            'frame_confidence': frame_detection['confidence_score'],
            'frame_detection_details': frame_detection['detection_details'],
            'final_shape': final_enhanced.shape,
            'replicate_used': frame_detection['has_black_frame'] and self.replicate_available
        }
        
        return final_enhanced, detected_color, processing_info

def handler(event):
    """RunPod handler function"""
    logger.info("=== Thumbnail Handler v38 Started ===")
    
    try:
        handler_instance = ThumbnailHandler()
        
        # Get input from event
        input_data = event.get('input', {})
        input_url = input_data.get('input_url')
        
        if not input_url:
            raise ValueError("No input_url provided")
        
        logger.info(f"Processing image from: {input_url}")
        
        # Process thumbnail
        processed_image, detected_color, processing_info = handler_instance.process_thumbnail(input_url)
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', processed_image, 
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        image_data_url = f"data:image/jpeg;base64,{image_base64}"
        
        result = {
            "output": {
                "thumbnail_image": image_data_url,
                "detected_color": detected_color,
                "processing_info": processing_info,
                "processing_time": datetime.now().isoformat(),
                "processing_method": "v38_enhanced_detection",
                "replicate_available": handler_instance.replicate_available,
                "replicate_used": processing_info.get('replicate_used', False),
                "success": True
            }
        }
        
        logger.info(f"Processing completed successfully")
        logger.info(f"Detected color: {detected_color}")
        logger.info(f"Black frame detected: {processing_info['has_black_frame']}")
        logger.info(f"Frame confidence: {processing_info['frame_confidence']:.2f}")
        
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

# For local testing
if __name__ == "__main__":
    test_event = {
        "input": {
            "input_url": "https://example.com/test-image.jpg"
        }
    }
    
    result = handler(test_event)
    print(json.dumps(result, indent=2))
