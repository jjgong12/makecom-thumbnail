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
VERSION = "v20-thumbnail"

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

class ThumbnailProcessorV20:
    """v20 Thumbnail Processor - Effective Black Box Detection"""
    
    def __init__(self):
        print(f"[{VERSION}] Initializing - Black Box Detection")
        self.replicate_client = None
    
    def detect_black_box(self, image_np):
        """검은 박스 감지 - 단순하지만 효과적"""
        h, w = image_np.shape[:2]
        print(f"[{VERSION}] Detecting black box in {w}x{h} image")
        
        # Grayscale 변환
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Method 1: 큰 검은 영역 직접 찾기
        # 여러 threshold 시도
        for thresh_val in [30, 40, 50, 60]:
            _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
            
            # 모폴로지 연산으로 노이즈 제거
            kernel = np.ones((5, 5), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Contour 찾기
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
                
            # 가장 큰 contour부터 확인
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            for contour in sorted_contours[:3]:  # 상위 3개만 확인
                x, y, cw, ch = cv2.boundingRect(contour)
                area = cw * ch
                
                # 조건 확인
                # 1. 충분히 큰 영역 (전체의 5% 이상)
                # 2. 대략 정사각형 (비율 0.7~1.3)
                # 3. 중앙 근처에 위치
                if area > (w * h * 0.05):
                    aspect_ratio = cw / ch
                    center_x = x + cw/2
                    center_y = y + ch/2
                    
                    # 중앙에서 너무 멀지 않은지 확인
                    dist_from_center = np.sqrt((center_x - w/2)**2 + (center_y - h/2)**2)
                    max_dist = min(w, h) * 0.3
                    
                    if 0.7 < aspect_ratio < 1.3 and dist_from_center < max_dist:
                        print(f"[{VERSION}] Black box found at ({x},{y}) size {cw}x{ch} with threshold {thresh_val}")
                        
                        # 정확한 마스크 생성
                        mask = np.zeros((h, w), dtype=np.uint8)
                        cv2.drawContours(mask, [contour], -1, 255, -1)
                        
                        return {
                            'has_frame': True,
                            'bbox': (x, y, cw, ch),
                            'mask': mask,
                            'method': f'threshold_{thresh_val}'
                        }
        
        # Method 2: 엣지 검사 (fallback)
        edge_thickness = self._check_edges(gray)
        if edge_thickness > 20:
            print(f"[{VERSION}] Edge frame detected: {edge_thickness}px")
            mask = self._create_edge_mask(h, w, edge_thickness)
            return {
                'has_frame': True,
                'thickness': edge_thickness,
                'mask': mask,
                'method': 'edge'
            }
        
        print(f"[{VERSION}] No black box detected")
        return {'has_frame': False, 'mask': None}
    
    def _check_edges(self, gray):
        """가장자리 검은 프레임 확인"""
        h, w = gray.shape
        threshold = 50
        
        # 각 가장자리 확인
        edges = {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
        
        # Top
        for i in range(min(200, h//3)):
            if np.mean(gray[i, :]) < threshold:
                edges['top'] = i + 1
            else:
                break
        
        # Bottom
        for i in range(min(200, h//3)):
            if np.mean(gray[h-1-i, :]) < threshold:
                edges['bottom'] = i + 1
            else:
                break
        
        # Left
        for i in range(min(200, w//3)):
            if np.mean(gray[:, i]) < threshold:
                edges['left'] = i + 1
            else:
                break
        
        # Right
        for i in range(min(200, w//3)):
            if np.mean(gray[:, w-1-i]) < threshold:
                edges['right'] = i + 1
            else:
                break
        
        # 평균 두께
        avg_thickness = np.mean(list(edges.values()))
        return int(avg_thickness) if avg_thickness > 20 else 0
    
    def _create_edge_mask(self, h, w, thickness):
        """엣지 마스크 생성"""
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[:thickness, :] = 255  # Top
        mask[-thickness:, :] = 255  # Bottom
        mask[:, :thickness] = 255  # Left
        mask[:, -thickness:] = 255  # Right
        return mask
    
    def remove_black_frame_replicate(self, image, frame_info):
        """Replicate API로 검은 프레임 제거"""
        if not frame_info['has_frame'] or not REPLICATE_AVAILABLE:
            # Fallback: 검은 영역 크롭
            if 'bbox' in frame_info:
                x, y, w, h = frame_info['bbox']
                return image.crop((x+10, y+10, x+w-10, y+h-10))
            return image
        
        try:
            print(f"[{VERSION}] Removing black frame with Replicate")
            
            # 마스크 확장
            mask_np = frame_info['mask']
            kernel = np.ones((15, 15), np.uint8)
            mask_np = cv2.dilate(mask_np, kernel, iterations=2)
            
            mask_img = Image.fromarray(mask_np)
            
            # Base64 인코딩
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            
            mask_buffer = io.BytesIO()
            mask_img.save(mask_buffer, format='PNG')
            mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode('utf-8')
            
            # Replicate 클라이언트
            if not self.replicate_client:
                self.replicate_client = replicate.Client(api_token=os.environ.get('REPLICATE_API_TOKEN'))
            
            # 인페인팅 실행
            output = self.replicate_client.run(
                "stability-ai/stable-diffusion-inpainting",
                input={
                    "image": f"data:image/png;base64,{img_base64}",
                    "mask": f"data:image/png;base64,{mask_base64}",
                    "prompt": "clean white background, product photography background, seamless",
                    "negative_prompt": "black, dark, frame, border, box, shadow",
                    "num_inference_steps": 35,
                    "guidance_scale": 9.0
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
        
        # Fallback
        if 'bbox' in frame_info:
            x, y, w, h = frame_info['bbox']
            return image.crop((x+10, y+10, x+w-10, y+h-10))
        
        return image
    
    def apply_simple_enhancement(self, image):
        """Enhancement와 동일한 간단한 색감 보정"""
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
        """웨딩링 중심으로 썸네일 생성 + 디테일 보정"""
        img_np = np.array(image)
        h, w = img_np.shape[:2]
        
        # 웨딩링 영역 찾기
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # 밝은 영역(링) 찾기
        _, bright = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        
        # 엣지 검출
        edges = cv2.Canny(gray, 50, 150)
        combined = cv2.bitwise_or(bright, edges)
        
        # Contour 찾기
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 모든 contour를 포함하는 바운딩 박스
            all_points = np.concatenate(contours)
            x, y, w_box, h_box = cv2.boundingRect(all_points)
            
            # 15% 패딩 추가
            padding_x = int(w_box * 0.15)
            padding_y = int(h_box * 0.15)
            
            x = max(0, x - padding_x)
            y = max(0, y - padding_y)
            w_box = min(w - x, w_box + 2 * padding_x)
            h_box = min(h - y, h_box + 2 * padding_y)
            
            # 크롭
            cropped = img_np[y:y+h_box, x:x+w_box]
        else:
            # Fallback: 중앙 크롭
            cropped = img_np
        
        # 타겟 크기로 리사이즈
        target_w, target_h = target_size
        h_crop, w_crop = cropped.shape[:2]
        
        # 비율 유지하며 fit
        scale = min(target_w / w_crop, target_h / h_crop)
        new_w = int(w_crop * scale)
        new_h = int(h_crop * scale)
        
        # 고품질 리사이즈
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # 캔버스 생성 및 중앙 배치
        canvas = np.full((target_h, target_w, 3), (245, 243, 240), dtype=np.uint8)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # PIL로 변환 후 디테일 보정
        thumb_img = Image.fromarray(canvas)
        
        # 선명도 증가
        enhancer = ImageEnhance.Sharpness(thumb_img)
        thumb_img = enhancer.enhance(1.4)
        
        # 엣지 강화
        thumb_np = np.array(thumb_img)
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]]) / 10.0
        
        sharpened = cv2.filter2D(thumb_np, -1, kernel)
        
        # 원본과 블렌딩
        result = cv2.addWeighted(thumb_np, 0.7, sharpened, 0.3, 0)
        
        print(f"[{VERSION}] Created {target_w}x{target_h} thumbnail")
        
        return Image.fromarray(result.astype(np.uint8))

# 전역 인스턴스
processor_instance = None

def get_processor():
    global processor_instance
    if processor_instance is None:
        processor_instance = ThumbnailProcessorV20()
    return processor_instance

def find_base64_in_dict(data, depth=0, max_depth=10):
    """중첩된 딕셔너리에서 base64 이미지 찾기"""
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
    """Base64 문자열을 PIL Image로 디코드"""
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
    """이미지를 base64로 인코딩"""
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        buffer = io.BytesIO()
        image.save(buffer, format=format, quality=95 if format == 'JPEG' else None)
        buffer.seek(0)
        
        # Base64 인코딩 - Make.com을 위해 padding 제거
        base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        base64_str = base64_str.rstrip('=')
        
        return base64_str
        
    except Exception as e:
        print(f"[{VERSION}] Error encoding image: {e}")
        raise

def handler(job):
    """RunPod 썸네일 핸들러"""
    try:
        start_time = time.time()
        job_input = job["input"]
        
        print(f"[{VERSION}] Thumbnail processing started")
        print(f"[{VERSION}] REPLICATE_AVAILABLE: {REPLICATE_AVAILABLE}")
        
        # Base64 이미지 찾기
        base64_image = find_base64_in_dict(job_input)
        if not base64_image:
            return {
                "output": {
                    "error": "No image data found",
                    "version": VERSION,
                    "success": False
                }
            }
        
        # 이미지 디코드
        image = decode_base64_image(base64_image)
        print(f"[{VERSION}] Image decoded: {image.size}")
        
        # numpy 변환
        image_np = np.array(image)
        
        # 1. 검은색 박스 감지
        processor = get_processor()
        frame_info = processor.detect_black_box(image_np)
        
        # 2. 프레임 제거
        if frame_info['has_frame']:
            print(f"[{VERSION}] Removing black box")
            image = processor.remove_black_frame_replicate(image, frame_info)
        
        # 3. 색감 보정
        image = processor.apply_simple_enhancement(image)
        
        # 4. 썸네일 생성 + 디테일 보정
        thumbnail = processor.create_thumbnail_with_detail(image, (1000, 1300))
        
        # 결과 인코딩
        thumbnail_base64 = encode_image_to_base64(thumbnail)
        
        # 처리 시간
        processing_time = time.time() - start_time
        print(f"[{VERSION}] Processing completed in {processing_time:.2f}s")
        
        # Return 구조
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

# RunPod 시작
if __name__ == "__main__":
    print("="*70)
    print(f"Wedding Ring Thumbnail {VERSION}")
    print("Thumbnail Handler (b_파일)")
    print(f"Replicate Available: {REPLICATE_AVAILABLE}")
    print("="*70)
    
    runpod.serverless.start({"handler": handler})
