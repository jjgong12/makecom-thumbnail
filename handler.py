import runpod
import base64
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import cv2
import io
import os
import traceback
import time

# Version info
VERSION = "v37-thumbnail"

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

class ThumbnailProcessorV37:
    """v37 Thumbnail Processor - Better Black Detection + Stricter Color"""
    
    def __init__(self):
        print(f"[{VERSION}] Initializing - Improved Black Detection & Color Classification")
        self.replicate_client = None
        if REPLICATE_AVAILABLE and os.environ.get('REPLICATE_API_TOKEN'):
            self.replicate_client = replicate.Client(api_token=os.environ['REPLICATE_API_TOKEN'])
    
    def detect_metal_color_v37(self, image):
        """v37: Much stricter color detection - yellow gold only for TRUE yellow"""
        try:
            img_np = np.array(image)
            h, w = img_np.shape[:2]
            
            # Get center region
            center_y, center_x = h//2, w//2
            crop_size = min(h, w) // 3
            
            y1 = max(0, center_y - crop_size)
            y2 = min(h, center_y + crop_size)
            x1 = max(0, center_x - crop_size)
            x2 = min(w, center_x + crop_size)
            
            center_region = img_np[y1:y2, x1:x2]
            
            # Calculate color statistics
            r_mean = np.mean(center_region[:, :, 0])
            g_mean = np.mean(center_region[:, :, 1])
            b_mean = np.mean(center_region[:, :, 2])
            
            brightness = (r_mean + g_mean + b_mean) / 3
            max_channel = max(r_mean, g_mean, b_mean)
            min_channel = min(r_mean, g_mean, b_mean)
            saturation = max_channel - min_channel
            
            # Calculate yellowness - how yellow is it really?
            yellowness = (r_mean + g_mean) / 2 - b_mean
            
            print(f"[{VERSION}] Color analysis - R:{r_mean:.1f} G:{g_mean:.1f} B:{b_mean:.1f}")
            print(f"[{VERSION}] Brightness:{brightness:.1f} Saturation:{saturation:.1f} Yellowness:{yellowness:.1f}")
            
            # Very strict detection - prioritize non-yellow colors
            # 1. Rose Gold - clear pinkish tone
            if r_mean - b_mean > 25 and r_mean > g_mean > b_mean and r_mean - g_mean > 10:
                print(f"[{VERSION}] Detected: Rose Gold")
                return "rose_gold"
            # 2. Yellow Gold - ONLY if clearly yellow (very strict)
            elif yellowness > 40 and g_mean > 180 and saturation > 30 and r_mean > 200 and g_mean > 190:
                print(f"[{VERSION}] Detected: Yellow Gold (true yellow)")
                return "yellow_gold"
            # 3. Plain White - very bright and colorless
            elif brightness > 235 and saturation < 8:
                print(f"[{VERSION}] Detected: Plain White")
                return "plain_white"
            # 4. White Gold - bright metallic but not pure white
            elif brightness > 190 and saturation < 25:
                print(f"[{VERSION}] Detected: White Gold")
                return "white_gold"
            # 5. Default to plain white (not yellow gold!)
            else:
                print(f"[{VERSION}] Detected: Plain White (default)")
                return "plain_white"
                
        except Exception as e:
            print(f"[{VERSION}] Error in metal detection: {e}")
            return "plain_white"  # Safe default
    
    def detect_and_remove_black_box_v37(self, image):
        """v37: Enhanced black box detection with multiple strategies"""
        try:
            img_np = np.array(image)
            h, w = img_np.shape[:2]
            print(f"[{VERSION}] Detecting black box in {w}x{h} image - v37 enhanced algorithm")
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
            # Strategy 1: Progressive threshold scanning
            black_detected = False
            final_edges = {'top': 0, 'bottom': h, 'left': 0, 'right': w}
            
            # Use very low thresholds to catch any dark areas
            for threshold in [15, 20, 25, 30, 35, 40, 45, 50]:
                edges = {'top': 0, 'bottom': h, 'left': 0, 'right': w}
                
                # Top edge - scan more rows
                for y in range(min(h//2, 1000)):
                    row = gray[y, w//4:3*w//4]
                    if len(row) > 0 and np.mean(row) > threshold:
                        edges['top'] = y
                        break
                    # Also check if mostly black pixels
                    black_pixels = np.sum(row < threshold)
                    if black_pixels < len(row) * 0.8:
                        edges['top'] = y
                        break
                
                # Bottom edge - scan more rows
                for y in range(min(h//2, 1000)):
                    row = gray[h-1-y, w//4:3*w//4]
                    if len(row) > 0 and np.mean(row) > threshold:
                        edges['bottom'] = h - y
                        break
                    black_pixels = np.sum(row < threshold)
                    if black_pixels < len(row) * 0.8:
                        edges['bottom'] = h - y
                        break
                
                # Left edge
                for x in range(min(w//2, 1000)):
                    col = gray[h//4:3*h//4, x]
                    if len(col) > 0 and np.mean(col) > threshold:
                        edges['left'] = x
                        break
                    black_pixels = np.sum(col < threshold)
                    if black_pixels < len(col) * 0.8:
                        edges['left'] = x
                        break
                
                # Right edge
                for x in range(min(w//2, 1000)):
                    col = gray[h//4:3*h//4, w-1-x]
                    if len(col) > 0 and np.mean(col) > threshold:
                        edges['right'] = w - x
                        break
                    black_pixels = np.sum(col < threshold)
                    if black_pixels < len(col) * 0.8:
                        edges['right'] = w - x
                        break
                
                # Update final edges with most aggressive detection
                if edges['top'] > final_edges['top']:
                    final_edges['top'] = edges['top']
                if edges['bottom'] < final_edges['bottom']:
                    final_edges['bottom'] = edges['bottom']
                if edges['left'] > final_edges['left']:
                    final_edges['left'] = edges['left']
                if edges['right'] < final_edges['right']:
                    final_edges['right'] = edges['right']
            
            # Strategy 2: Check image borders directly
            border_size = 50
            
            # Check if borders are mostly black
            top_border = gray[:border_size, :]
            if np.mean(top_border) < 30:
                black_detected = True
                final_edges['top'] = max(final_edges['top'], border_size)
            
            bottom_border = gray[-border_size:, :]
            if np.mean(bottom_border) < 30:
                black_detected = True
                final_edges['bottom'] = min(final_edges['bottom'], h - border_size)
            
            left_border = gray[:, :border_size]
            if np.mean(left_border) < 30:
                black_detected = True
                final_edges['left'] = max(final_edges['left'], border_size)
            
            right_border = gray[:, -border_size:]
            if np.mean(right_border) < 30:
                black_detected = True
                final_edges['right'] = min(final_edges['right'], w - border_size)
            
            # Strategy 3: Global histogram check
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            dark_pixels = np.sum(hist[:40])  # Pixels with value < 40
            total_pixels = w * h
            dark_ratio = dark_pixels / total_pixels
            
            if dark_ratio > 0.1:  # If more than 10% dark pixels
                black_detected = True
            
            # Check if we found any black frame
            frame_found = (final_edges['top'] > 20 or 
                          final_edges['bottom'] < h - 20 or 
                          final_edges['left'] > 20 or 
                          final_edges['right'] < w - 20 or
                          black_detected)
            
            if frame_found:
                print(f"[{VERSION}] Black frame DETECTED! T:{final_edges['top']}, B:{final_edges['bottom']}, L:{final_edges['left']}, R:{final_edges['right']}")
                
                # Add safety margin and ensure minimum crop
                margin = 30
                final_edges['top'] = max(0, final_edges['top'] - margin)
                final_edges['bottom'] = min(h, final_edges['bottom'] + margin)
                final_edges['left'] = max(0, final_edges['left'] - margin)
                final_edges['right'] = min(w, final_edges['right'] + margin)
                
                # Ensure we have something to crop
                if final_edges['right'] - final_edges['left'] > 100 and final_edges['bottom'] - final_edges['top'] > 100:
                    cropped = img_np[final_edges['top']:final_edges['bottom'], 
                                   final_edges['left']:final_edges['right']]
                    
                    # If Replicate is available and significant black area, use it
                    if self.replicate_client and dark_ratio > 0.15:
                        print(f"[{VERSION}] Using Replicate for black box removal")
                        return self.remove_black_box_replicate(image, final_edges), True
                    else:
                        return Image.fromarray(cropped), True
            
            print(f"[{VERSION}] No black frame detected (dark_ratio: {dark_ratio:.2f})")
            return image, False
            
        except Exception as e:
            print(f"[{VERSION}] Error in black frame detection: {e}")
            traceback.print_exc()
            return image, False
    
    def remove_black_box_replicate(self, image, edges):
        """Use Replicate API to inpaint black areas"""
        try:
            # Create mask for black areas
            img_np = np.array(image)
            h, w = img_np.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # Mark black areas with some expansion
            expansion = 20
            if edges['top'] > 0:
                mask[:min(edges['top'] + expansion, h), :] = 255
            if edges['bottom'] < h:
                mask[max(edges['bottom'] - expansion, 0):, :] = 255
            if edges['left'] > 0:
                mask[:, :min(edges['left'] + expansion, w)] = 255
            if edges['right'] < w:
                mask[:, max(edges['right'] - expansion, 0):] = 255
            
            # Convert to PIL
            mask_img = Image.fromarray(mask)
            
            # Prepare for Replicate
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            mask_buffer = io.BytesIO()
            mask_img.save(mask_buffer, format='PNG')
            mask_buffer.seek(0)
            
            # Run inpainting
            output = self.replicate_client.run(
                "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3",
                input={
                    "image": img_buffer,
                    "mask": mask_buffer,
                    "prompt": "clean white background, product photography background, seamless white, no shadows",
                    "negative_prompt": "black frame, black border, dark edges, shadows, dark areas",
                    "num_inference_steps": 35,
                    "guidance_scale": 12
                }
            )
            
            # Get result
            if output and len(output) > 0:
                response = requests.get(output[0])
                result_img = Image.open(io.BytesIO(response.content))
                return result_img
            else:
                # Fallback to cropping
                cropped = img_np[edges['top']:edges['bottom'], 
                               edges['left']:edges['right']]
                return Image.fromarray(cropped)
                
        except Exception as e:
            print(f"[{VERSION}] Error in Replicate inpainting: {e}")
            # Fallback to cropping
            img_np = np.array(image)
            cropped = img_np[edges['top']:edges['bottom'], 
                           edges['left']:edges['right']]
            return Image.fromarray(cropped)
    
    def remove_noise_and_defects(self, image):
        """Advanced noise, dust, and scratch removal"""
        try:
            img_np = np.array(image)
            
            # 1. Initial denoising
            denoised = cv2.fastNlMeansDenoisingColored(img_np, None, 3, 3, 7, 21)
            
            # 2. Dust removal
            dust_removed = cv2.medianBlur(denoised, 3)
            
            # 3. Bilateral filter
            smooth = cv2.bilateralFilter(dust_removed, 5, 30, 30)
            
            # 4. Scratch detection and removal
            gray = cv2.cvtColor(smooth, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 30, 60)
            
            kernel_line = np.ones((3,1), np.uint8)
            scratches_v = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_line, iterations=1)
            kernel_line = np.ones((1,3), np.uint8)
            scratches_h = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_line, iterations=1)
            scratches = cv2.bitwise_or(scratches_v, scratches_h)
            
            scratches = cv2.dilate(scratches, np.ones((3,3), np.uint8), iterations=1)
            
            if np.any(scratches):
                result = cv2.inpaint(smooth, scratches, 3, cv2.INPAINT_TELEA)
            else:
                result = smooth
            
            # 5. Final smoothing
            result = cv2.GaussianBlur(result, (3, 3), 0.5)
            result = cv2.addWeighted(img_np, 0.3, result, 0.7, 0)
            
            return Image.fromarray(result)
            
        except Exception as e:
            print(f"[{VERSION}] Error in noise/defect removal: {e}")
            return image
    
    def apply_color_aware_enhancement_v37(self, image, metal_type):
        """v37: Apply enhancement based on detected metal color"""
        try:
            # First apply noise and defect removal
            image = self.remove_noise_and_defects(image)
            
            # Metal-specific enhancement parameters
            if metal_type == "yellow_gold":
                # True yellow gold - moderate enhancement
                sharpness_radius = 1.5
                sharpness_percent = 95
                brightness = 1.14
                contrast = 1.10
                saturation = 1.05
                gamma = 0.92
                detail_boost = 45
            elif metal_type == "plain_white":
                # Extra bright for plain white
                sharpness_radius = 1.8
                sharpness_percent = 125
                brightness = 1.25
                contrast = 1.12
                saturation = 0.90
                gamma = 0.83
                detail_boost = 65
            elif metal_type == "rose_gold":
                # Warm enhancement for rose gold
                sharpness_radius = 1.5
                sharpness_percent = 105
                brightness = 1.16
                contrast = 1.10
                saturation = 1.06
                gamma = 0.89
                detail_boost = 52
            elif metal_type == "white_gold":
                # Cool enhancement for white gold
                sharpness_radius = 1.6
                sharpness_percent = 115
                brightness = 1.20
                contrast = 1.12
                saturation = 0.93
                gamma = 0.86
                detail_boost = 58
            else:
                # Default to plain white settings
                sharpness_radius = 1.8
                sharpness_percent = 125
                brightness = 1.25
                contrast = 1.12
                saturation = 0.90
                gamma = 0.83
                detail_boost = 65
            
            # 1. Strong sharpening for detail
            image = image.filter(ImageFilter.UnsharpMask(
                radius=sharpness_radius, 
                percent=sharpness_percent, 
                threshold=2
            ))
            
            # 2. Brightness
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness)
            
            # 3. Contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast)
            
            # 4. Saturation
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(saturation)
            
            # 5. Convert to numpy for advanced processing
            img_np = np.array(image)
            h, w = img_np.shape[:2]
            
            # 6. High-pass filter for detail enhancement
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (21, 21), 0)
            high_pass = cv2.subtract(gray, blur)
            high_pass = cv2.normalize(high_pass, None, 0, detail_boost, cv2.NORM_MINMAX)
            
            for i in range(3):
                img_np[:, :, i] = cv2.add(img_np[:, :, i], high_pass)
            
            # 7. White background
            white_color = (254, 254, 254)
            edges = cv2.Canny(gray, 60, 150)
            edges_dilated = cv2.dilate(edges, np.ones((5,5), np.uint8), iterations=2)
            
            mask = np.ones((h, w), dtype=np.float32)
            mask[edges_dilated > 0] = 0
            mask = cv2.GaussianBlur(mask, (51, 51), 25)
            
            for i in range(3):
                img_np[:, :, i] = img_np[:, :, i] * (1 - mask * 0.25) + white_color[i] * mask * 0.25
            
            # 8. Gamma correction
            img_np = np.power(img_np / 255.0, gamma) * 255
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            
            # 9. CLAHE for micro-contrast
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            img_np = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
            
            return Image.fromarray(img_np)
            
        except Exception as e:
            print(f"[{VERSION}] Error in enhancement: {e}")
            return image
    
    def create_perfect_thumbnail_1000x1300(self, image):
        """Create perfect 1000x1300 thumbnail with tight crop"""
        try:
            target_size = (1000, 1300)
            img_np = np.array(image)
            h, w = img_np.shape[:2]
            
            print(f"[{VERSION}] Creating tight crop thumbnail...")
            
            # Tight crop - 28% for extra tight framing (even tighter than v36)
            crop_ratio = 1.3
            
            if w / h > 1 / crop_ratio:
                crop_h = int(h * 0.28)
                crop_w = int(crop_h / crop_ratio)
            else:
                crop_w = int(w * 0.28)
                crop_h = int(crop_w * crop_ratio)
            
            # Center the crop
            x = (w - crop_w) // 2
            y = (h - crop_h) // 2
            
            # Ensure bounds are valid
            x = max(0, x)
            y = max(0, y)
            crop_w = min(crop_w, w - x)
            crop_h = min(crop_h, h - y)
            
            print(f"[{VERSION}] Extra tight crop: ({x},{y}) size {crop_w}x{crop_h}")
            
            # Crop the image
            cropped = image.crop((x, y, x + crop_w, y + crop_h))
            
            # Resize to target size with high quality
            final = cropped.resize(target_size, Image.Resampling.LANCZOS)
            
            # Apply super-resolution-like enhancement
            final = self.super_resolution_enhance(final)
            
            print(f"[{VERSION}] Created 1000x1300 thumbnail")
            return final
            
        except Exception as e:
            print(f"[{VERSION}] Error creating thumbnail: {e}")
            traceback.print_exc()
            return image.resize((1000, 1300), Image.Resampling.LANCZOS)
    
    def super_resolution_enhance(self, image):
        """Apply super-resolution-like enhancement"""
        try:
            img_np = np.array(image)
            
            # 1. Edge-aware enhancement
            shifts = [(0,0), (1,0), (0,1), (1,1)]
            enhanced = np.zeros_like(img_np, dtype=np.float32)
            
            for dx, dy in shifts:
                shifted = np.roll(np.roll(img_np, dx, axis=1), dy, axis=0)
                enhanced += shifted.astype(np.float32)
            
            enhanced /= len(shifts)
            
            # 2. Laplacian edge enhancement
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian = np.abs(laplacian)
            laplacian = cv2.GaussianBlur(laplacian, (3, 3), 0)
            laplacian = np.clip(laplacian * 2, 0, 50)
            
            for i in range(3):
                enhanced[:, :, i] += laplacian
            
            # 3. Frequency enhancement
            for i in range(3):
                channel = enhanced[:, :, i]
                blur = cv2.GaussianBlur(channel, (5, 5), 0)
                detail = channel - blur
                enhanced[:, :, i] = channel + detail * 0.5
            
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
            
            # 4. Final sharpening
            enhanced = Image.fromarray(enhanced)
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(1.3)
            
            return enhanced
            
        except Exception as e:
            print(f"[{VERSION}] Error in super-resolution: {e}")
            return image

def find_base64_in_dict(data, depth=0, max_depth=10):
    """Find base64 image in nested dictionary"""
    if depth > max_depth:
        return None
    
    if isinstance(data, str) and len(data) > 100:
        return data
    
    if isinstance(data, dict):
        for key in ['image_base64', 'image', 'base64', 'data', 'input', 'file', 'imageData']:
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
        # Handle data URL format
        if ',' in base64_str and base64_str.startswith('data:'):
            base64_str = base64_str.split(',')[1]
        
        # Clean base64
        base64_str = base64_str.strip()
        
        # Add padding for decoding
        padding = 4 - len(base64_str) % 4
        if padding != 4:
            base64_str += '=' * padding
        
        # Decode
        img_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_data))
        
        # Convert to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        return img
        
    except Exception as e:
        print(f"[{VERSION}] Error decoding base64: {e}")
        raise

def encode_image_to_base64(image, format='PNG'):
    """Encode image to base64 (Make.com compatible)"""
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        buffer = io.BytesIO()
        image.save(buffer, format=format, quality=95 if format == 'JPEG' else None)
        buffer.seek(0)
        
        # Base64 encode
        base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # CRITICAL: Remove padding for Make.com
        base64_str = base64_str.rstrip('=')
        
        return base64_str
        
    except Exception as e:
        print(f"[{VERSION}] Error encoding image: {e}")
        raise

def handler(job):
    """RunPod handler - V37 with better black detection"""
    print(f"[{VERSION}] ====== Thumbnail Handler Started ======")
    print(f"[{VERSION}] Enhanced Black Detection & Stricter Color Classification")
    
    start_time = time.time()
    
    try:
        job_input = job.get("input", {})
        print(f"[{VERSION}] Input type: {type(job_input)}")
        print(f"[{VERSION}] Input keys: {list(job_input.keys()) if isinstance(job_input, dict) else 'Not a dict'}")
        
        # Find base64 image
        base64_image = find_base64_in_dict(job_input)
        
        if not base64_image:
            # Try direct string
            if isinstance(job_input, str) and len(job_input) > 100:
                base64_image = job_input
            else:
                return {
                    "output": {
                        "thumbnail": None,
                        "error": "No image data found",
                        "success": False,
                        "version": VERSION,
                        "debug_info": {
                            "input_keys": list(job_input.keys()) if isinstance(job_input, dict) else [],
                            "input_length": len(str(job_input))
                        }
                    }
                }
        
        print(f"[{VERSION}] Base64 image found, length: {len(base64_image)}")
        
        # Decode image
        try:
            image = decode_base64_image(base64_image)
            print(f"[{VERSION}] Image decoded: {image.size}")
        except Exception as e:
            return {
                "output": {
                    "thumbnail": None,
                    "error": f"Failed to decode image: {str(e)}",
                    "success": False,
                    "version": VERSION
                }
            }
        
        # Create processor
        processor = ThumbnailProcessorV37()
        
        # Process image step by step
        had_black_frame = False
        
        # 1. Detect and remove black box with v37 algorithm
        try:
            image, had_black_frame = processor.detect_and_remove_black_box_v37(image)
            print(f"[{VERSION}] v37 black box detection complete: {had_black_frame}")
        except Exception as e:
            print(f"[{VERSION}] Error in black frame detection: {e}")
            traceback.print_exc()
        
        # 2. Detect metal color with stricter algorithm
        try:
            metal_type = processor.detect_metal_color_v37(image)
            print(f"[{VERSION}] Detected metal type: {metal_type}")
        except Exception as e:
            print(f"[{VERSION}] Error in metal detection: {e}")
            metal_type = "plain_white"
        
        # 3. Apply color-aware enhancement - AFTER black box removal
        try:
            image = processor.apply_color_aware_enhancement_v37(image, metal_type)
            print(f"[{VERSION}] Color-aware enhancement applied for {metal_type}")
        except Exception as e:
            print(f"[{VERSION}] Error in enhancement: {e}")
            traceback.print_exc()
        
        # 4. Create PERFECT 1000x1300 thumbnail with tight crop
        try:
            thumbnail = processor.create_perfect_thumbnail_1000x1300(image)
            print(f"[{VERSION}] Perfect thumbnail created: {thumbnail.size}")
        except Exception as e:
            print(f"[{VERSION}] Error creating thumbnail: {e}")
            traceback.print_exc()
            thumbnail = image.resize((1000, 1300), Image.Resampling.LANCZOS)
        
        # Encode result
        try:
            thumbnail_base64 = encode_image_to_base64(thumbnail, format='PNG')
            print(f"[{VERSION}] Thumbnail encoded, length: {len(thumbnail_base64)}")
        except Exception as e:
            return {
                "output": {
                    "thumbnail": None,
                    "error": f"Failed to encode thumbnail: {str(e)}",
                    "success": False,
                    "version": VERSION
                }
            }
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Return proper structure
        result = {
            "output": {
                "thumbnail": thumbnail_base64,
                "has_black_frame": had_black_frame,
                "success": True,
                "version": VERSION,
                "thumbnail_size": [1000, 1300],
                "processing_method": "v37_enhanced_detection",
                "detected_metal": metal_type,
                "processing_time": round(processing_time, 2),
                "replicate_available": REPLICATE_AVAILABLE,
                "replicate_used": REPLICATE_AVAILABLE and had_black_frame,
                "enhancements_applied": [
                    "v37_black_box_detection",
                    "stricter_metal_color_detection",
                    "noise_removal",
                    "dust_scratch_removal",
                    f"{metal_type}_specific_enhancement",
                    "tight_crop_28_percent",
                    "super_resolution_simulation",
                    "clahe_micro_contrast"
                ],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                "warning": "Google Script must add padding: while (base64Data.length % 4 !== 0) { base64Data += '='; }"
            }
        }
        
        print(f"[{VERSION}] ====== Success - Returning v37 Thumbnail ======")
        print(f"[{VERSION}] Total processing time: {processing_time:.2f}s")
        print(f"[{VERSION}] Black frame detected and removed: {had_black_frame}")
        print(f"[{VERSION}] Metal type: {metal_type}")
        
        return result
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"[{VERSION}] CRITICAL ERROR: {error_msg}")
        traceback.print_exc()
        
        return {
            "output": {
                "thumbnail": None,
                "error": error_msg,
                "success": False,
                "version": VERSION,
                "traceback": traceback.format_exc()
            }
        }

# RunPod serverless start
if __name__ == "__main__":
    print("="*70)
    print(f"Wedding Ring Thumbnail {VERSION}")
    print("V37 - Enhanced Black Detection & Stricter Color")
    print("Features:")
    print("- Much stricter yellow gold detection")
    print("- Default to plain white (not yellow gold)")
    print("- Enhanced black box detection (multiple strategies)")
    print("- Lower thresholds for black detection")
    print("- Border checking for black frames")
    print("- Tighter crop (28%)")
    print("- Metal-specific enhancement parameters")
    print("- Advanced noise and defect removal")
    print("- Super-resolution-like enhancement")
    print("- CLAHE for local contrast")
    print(f"Replicate Available: {REPLICATE_AVAILABLE}")
    print("CRITICAL: Padding is removed for Make.com")
    print("Google Apps Script MUST add padding back:")
    print("while (base64Data.length % 4 !== 0) { base64Data += '='; }")
    print("="*70)
    
    runpod.serverless.start({"handler": handler})
