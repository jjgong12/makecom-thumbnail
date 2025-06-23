import runpod
import base64
import numpy as np
from PIL import Image, ImageEnhance
import cv2
import io
import os
import traceback
import requests
import time

# Version info
VERSION = "v18-thumbnail"

# Replicate API settings
REPLICATE_API_TOKEN = os.environ.get('REPLICATE_API_TOKEN')
if not REPLICATE_API_TOKEN:
    raise ValueError("REPLICATE_API_TOKEN not set")

class BlackBoxDetectorV18:
    """Enhanced black box detection for high-resolution images"""
    
    def __init__(self):
        print(f"[{VERSION}] BlackBoxDetectorV18 initialized")
        
    def detect_black_frame_multi_stage(self, image):
        """다단계 검증으로 검은 박스 감지 - 6720x4480 대응"""
        img_np = np.array(image)
        h, w = img_np.shape[:2]
        print(f"[{VERSION}] Detecting black frame in {w}x{h} image")
        
        # Stage 1: Edge-based detection (가장자리부터 안쪽으로)
        edge_results = self._detect_edges_inward(img_np)
        
        # Stage 2: Threshold-based detection (여러 임계값)
        threshold_results = self._detect_with_thresholds(img_np)
        
        # Stage 3: Gradient-based detection (경계선 감지)
        gradient_results = self._detect_with_gradients(img_np)
        
        # Stage 4: Color histogram analysis
        histogram_results = self._analyze_color_histogram(img_np)
        
        # Stage 5: Connected components analysis
        component_results = self._analyze_connected_components(img_np)
        
        # Combine all results
        all_results = [edge_results, threshold_results, gradient_results, 
                      histogram_results, component_results]
        
        # Vote on best detection
        final_result = self._vote_on_detection(all_results)
        
        if final_result['detected']:
            print(f"[{VERSION}] Black frame detected: thickness={final_result['thickness']}")
        else:
            print(f"[{VERSION}] No black frame detected")
            
        return final_result
    
    def _detect_edges_inward(self, img_np):
        """가장자리부터 안쪽으로 검사 - v16 방식 개선"""
        h, w = img_np.shape[:2]
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # 최대 검사 깊이 (고해상도 대응)
        max_depth = min(300, h//10, w//10)
        
        edges = {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
        threshold = 40  # 검은색 임계값
        
        # Top edge
        for y in range(max_depth):
            row = gray[y, :]
            if np.mean(row) < threshold:
                edges['top'] = y + 1
            else:
                break
        
        # Bottom edge
        for y in range(max_depth):
            row = gray[h-1-y, :]
            if np.mean(row) < threshold:
                edges['bottom'] = y + 1
            else:
                break
        
        # Left edge
        for x in range(max_depth):
            col = gray[:, x]
            if np.mean(col) < threshold:
                edges['left'] = x + 1
            else:
                break
        
        # Right edge
        for x in range(max_depth):
            col = gray[:, w-1-x]
            if np.mean(col) < threshold:
                edges['right'] = x + 1
            else:
                break
        
        avg_thickness = np.mean(list(edges.values()))
        detected = avg_thickness > 20  # 최소 20픽셀
        
        return {
            'detected': detected,
            'thickness': int(avg_thickness),
            'edges': edges,
            'method': 'edge_inward'
        }
    
    def _detect_with_thresholds(self, img_np):
        """여러 임계값으로 검사"""
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        best_result = {'detected': False, 'thickness': 0}
        
        # 다양한 임계값 시도
        for thresh in [20, 30, 40, 50, 60]:
            _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
            
            # Find largest black region
            contours, _ = cv2.findContours(255 - binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest = max(contours, key=cv2.contourArea)
                x, y, cw, ch = cv2.boundingRect(largest)
                
                # Check if it's a frame (touches edges)
                if x <= 10 and y <= 10 and x + cw >= w - 10 and y + ch >= h - 10:
                    # Calculate frame thickness
                    thickness = min(x, y, w - (x + cw), h - (y + ch))
                    if thickness > best_result['thickness']:
                        best_result = {
                            'detected': True,
                            'thickness': thickness,
                            'bbox': (x, y, cw, ch),
                            'method': f'threshold_{thresh}'
                        }
        
        return best_result
    
    def _detect_with_gradients(self, img_np):
        """경계선 감지를 통한 박스 검출"""
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Sobel gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Find strong edges
        _, edges = cv2.threshold(grad_mag.astype(np.uint8), 50, 255, cv2.THRESH_BINARY)
        
        # Hough lines to find box edges
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        if lines is not None:
            # Analyze lines to find box structure
            h, w = gray.shape
            h_lines = []
            v_lines = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y2 - y1) < 10:  # Horizontal
                    h_lines.append((y1 + y2) // 2)
                elif abs(x2 - x1) < 10:  # Vertical
                    v_lines.append((x1 + x2) // 2)
            
            # Find frame boundaries
            if h_lines and v_lines:
                top = min([y for y in h_lines if y < h//3])
                bottom = max([y for y in h_lines if y > 2*h//3])
                left = min([x for x in v_lines if x < w//3])
                right = max([x for x in v_lines if x > 2*w//3])
                
                thickness = min(top, left, w - right, h - bottom)
                
                if thickness > 20:
                    return {
                        'detected': True,
                        'thickness': thickness,
                        'method': 'gradient'
                    }
        
        return {'detected': False, 'thickness': 0}
    
    def _analyze_color_histogram(self, img_np):
        """색상 히스토그램 분석"""
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # Check for peak in dark values
        dark_pixels = np.sum(hist[:50])
        total_pixels = img_np.shape[0] * img_np.shape[1]
        dark_ratio = dark_pixels / total_pixels
        
        if dark_ratio > 0.15:  # 15% 이상이 어두운 픽셀
            # Estimate frame thickness based on dark pixel distribution
            h, w = gray.shape
            mask = gray < 50
            
            # Check edges
            top_dark = np.mean(mask[:100, :])
            bottom_dark = np.mean(mask[-100:, :])
            left_dark = np.mean(mask[:, :100])
            right_dark = np.mean(mask[:, -100:])
            
            if min(top_dark, bottom_dark, left_dark, right_dark) > 0.8:
                # Estimate thickness
                thickness = int(dark_ratio * min(h, w) * 0.3)
                return {
                    'detected': True,
                    'thickness': thickness,
                    'method': 'histogram'
                }
        
        return {'detected': False, 'thickness': 0}
    
    def _analyze_connected_components(self, img_np):
        """연결된 구성 요소 분석"""
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)
        
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        
        h, w = gray.shape
        frame_candidates = []
        
        for i in range(1, num_labels):
            x, y, width, height, area = stats[i]
            
            # Check if component could be a frame
            if (x <= 10 and y <= 10 and 
                x + width >= w - 10 and y + height >= h - 10 and
                area > 0.3 * w * h):  # Large area
                
                # This might be our black frame
                # Calculate thickness
                mask = (labels == i).astype(np.uint8)
                thickness = self._calculate_frame_thickness(mask)
                
                if thickness > 20:
                    frame_candidates.append({
                        'detected': True,
                        'thickness': thickness,
                        'method': 'components'
                    })
        
        if frame_candidates:
            return max(frame_candidates, key=lambda x: x['thickness'])
        
        return {'detected': False, 'thickness': 0}
    
    def _calculate_frame_thickness(self, mask):
        """프레임 두께 계산"""
        h, w = mask.shape
        
        # Find inner rectangle
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        
        # Find inner hole
        inner_mask = mask[ymin:ymax+1, xmin:xmax+1]
        inner_inv = 1 - inner_mask
        
        if np.any(inner_inv):
            inner_rows = np.any(inner_inv, axis=1)
            inner_cols = np.any(inner_inv, axis=0)
            
            if np.any(inner_rows) and np.any(inner_cols):
                iymin, iymax = np.where(inner_rows)[0][[0, -1]]
                ixmin, ixmax = np.where(inner_cols)[0][[0, -1]]
                
                # Calculate thickness
                thickness = min(
                    iymin, ixmin,
                    inner_mask.shape[0] - iymax - 1,
                    inner_mask.shape[1] - ixmax - 1
                )
                
                return thickness
        
        return 0
    
    def _vote_on_detection(self, results):
        """모든 검출 결과를 종합하여 최종 결정"""
        valid_results = [r for r in results if r['detected']]
        
        if not valid_results:
            return {'detected': False, 'thickness': 0, 'has_black_frame': False}
        
        # Calculate average thickness
        thicknesses = [r['thickness'] for r in valid_results]
        avg_thickness = int(np.mean(thicknesses))
        
        # Need at least 2 methods to agree
        if len(valid_results) >= 2:
            return {
                'detected': True,
                'thickness': avg_thickness,
                'has_black_frame': True,
                'confidence': len(valid_results) / len(results),
                'methods': [r['method'] for r in valid_results]
            }
        
        return {'detected': False, 'thickness': 0, 'has_black_frame': False}


class ThumbnailProcessorV18:
    """Thumbnail processing with advanced black box removal"""
    
    def __init__(self):
        self.detector = BlackBoxDetectorV18()
        print(f"[{VERSION}] ThumbnailProcessorV18 initialized")
    
    def remove_black_frame_replicate(self, image, detection_result):
        """Replicate API를 사용한 마스킹 제거"""
        if not detection_result['detected']:
            return image
        
        try:
            import replicate
            
            # Convert image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Prepare mask based on detection
            mask = self._create_mask_from_detection(image, detection_result)
            mask_buffered = io.BytesIO()
            mask.save(mask_buffered, format="PNG")
            mask_str = base64.b64encode(mask_buffered.getvalue()).decode()
            
            # Call Replicate API
            output = replicate.run(
                "andreasjansson/stable-diffusion-inpainting:8eb2da8345bee796efcd925573f077e36ed5fb4ea3ba240ef70c23cf33f0d848",
                input={
                    "image": f"data:image/png;base64,{img_str}",
                    "mask": f"data:image/png;base64,{mask_str}",
                    "prompt": "clean white background, product photography background",
                    "negative_prompt": "black frame, black border, dark edges",
                    "num_inference_steps": 25,
                    "guidance_scale": 7.5
                }
            )
            
            # Process result
            if output and len(output) > 0:
                response = requests.get(output[0])
                return Image.open(io.BytesIO(response.content))
            
        except Exception as e:
            print(f"[{VERSION}] Replicate API error: {e}")
        
        # Fallback to local removal
        return self._remove_frame_local(image, detection_result)
    
    def _create_mask_from_detection(self, image, detection_result):
        """검출 결과를 바탕으로 마스크 생성"""
        w, h = image.size
        mask = Image.new('L', (w, h), 0)
        
        if 'edges' in detection_result:
            edges = detection_result['edges']
            # Draw white rectangles for frame areas
            from PIL import ImageDraw
            draw = ImageDraw.Draw(mask)
            
            # Top
            if edges['top'] > 0:
                draw.rectangle([0, 0, w, edges['top']], fill=255)
            # Bottom
            if edges['bottom'] > 0:
                draw.rectangle([0, h - edges['bottom'], w, h], fill=255)
            # Left
            if edges['left'] > 0:
                draw.rectangle([0, 0, edges['left'], h], fill=255)
            # Right
            if edges['right'] > 0:
                draw.rectangle([w - edges['right'], 0, w, h], fill=255)
        else:
            # Use thickness for uniform frame
            t = detection_result['thickness']
            from PIL import ImageDraw
            draw = ImageDraw.Draw(mask)
            draw.rectangle([0, 0, w, t], fill=255)  # Top
            draw.rectangle([0, h-t, w, h], fill=255)  # Bottom
            draw.rectangle([0, 0, t, h], fill=255)  # Left
            draw.rectangle([w-t, 0, w, h], fill=255)  # Right
        
        return mask
    
    def _remove_frame_local(self, image, detection_result):
        """로컬 방식으로 프레임 제거"""
        img_np = np.array(image)
        
        if 'edges' in detection_result:
            edges = detection_result['edges']
            # Crop out the frame
            img_np = img_np[
                edges['top']:img_np.shape[0]-edges['bottom'],
                edges['left']:img_np.shape[1]-edges['right']
            ]
        else:
            # Use uniform thickness
            t = detection_result['thickness']
            img_np = img_np[t:-t, t:-t]
        
        return Image.fromarray(img_np)
    
    def apply_simple_enhancement(self, image):
        """색감 보정 - v16과 동일"""
        # 1. 밝기
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.1)
        
        # 2. 대비
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.05)
        
        # 3. 채도
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.02)
        
        # 4. 배경색 블렌딩
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
        """크롭 후 디테일 보정 추가"""
        img_np = np.array(image)
        h, w = img_np.shape[:2]
        target_w, target_h = target_size
        
        # 비율 계산
        target_ratio = target_w / target_h
        current_ratio = w / h
        
        # 크롭
        if current_ratio > target_ratio:
            new_w = int(h * target_ratio)
            crop_x = (w - new_w) // 2
            cropped = img_np[:, crop_x:crop_x + new_w]
        else:
            new_h = int(w / target_ratio)
            crop_y = (h - new_h) // 2
            cropped = img_np[crop_y:crop_y + new_h, :]
        
        # 리사이즈
        thumbnail = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        
        # PIL로 변환해서 디테일 보정
        thumb_img = Image.fromarray(thumbnail)
        
        # 디테일 강화 (썸네일이 확대되었으므로)
        enhancer = ImageEnhance.Sharpness(thumb_img)
        thumb_img = enhancer.enhance(1.3)  # 선명도 증가
        
        # 엣지 강화
        thumb_np = np.array(thumb_img)
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]]) / 9.0
        
        sharpened = cv2.filter2D(thumb_np, -1, kernel)
        
        # 원본과 블렌딩 (너무 과하지 않게)
        result = cv2.addWeighted(thumb_np, 0.7, sharpened, 0.3, 0)
        
        return Image.fromarray(result)


def handler(job):
    """RunPod handler function - Thumbnail processing"""
    print(f"[{VERSION}] Handler started")
    job_input = job['input']
    
    try:
        # Get image data
        if 'image' not in job_input:
            raise ValueError("No 'image' field in input")
        
        image_data = job_input['image']
        
        # Decode base64 image
        if image_data.startswith('data:'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        print(f"[{VERSION}] Image loaded: {image.size}")
        
        # Create processor instance
        processor = ThumbnailProcessorV18()
        
        # 1. Detect black frame with multi-stage validation
        detection_result = processor.detector.detect_black_frame_multi_stage(image)
        
        # 2. Remove black frame if detected
        if detection_result['detected']:
            image = processor.remove_black_frame_replicate(image, detection_result)
            print(f"[{VERSION}] Black frame removed")
        
        # 3. Apply color enhancement
        image = processor.apply_simple_enhancement(image)
        
        # 4. Create thumbnail with detail enhancement
        thumbnail = processor.create_thumbnail_with_detail(image)
        
        # Convert to base64
        buffered = io.BytesIO()
        thumbnail.save(buffered, format="JPEG", quality=95)
        thumbnail_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # IMPORTANT: Remove padding for Make.com
        thumbnail_base64 = thumbnail_base64.rstrip('=')
        
        # Return with proper structure for Make.com
        return {
            "output": {
                "thumbnail": f"data:image/jpeg;base64,{thumbnail_base64}",
                "has_black_frame": detection_result.get('has_black_frame', False),
                "frame_thickness": detection_result.get('thickness', 0),
                "detection_confidence": detection_result.get('confidence', 0),
                "detection_methods": detection_result.get('methods', []),
                "status": "success",
                "version": VERSION,
                "processing_time": time.time() - job.get('start_time', time.time())
            }
        }
        
    except Exception as e:
        error_msg = f"Error in thumbnail processing: {str(e)}\n{traceback.format_exc()}"
        print(f"[{VERSION}] {error_msg}")
        
        return {
            "output": {
                "error": error_msg,
                "status": "error",
                "version": VERSION
            }
        }

# RunPod serverless handler
runpod.serverless.start({"handler": handler})
