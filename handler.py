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
VERSION = "v16-thumbnail"

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

class ThumbnailProcessorV16:
    """v16 Thumbnail Processor - Improved Detection"""
    
    def __init__(self):
        print(f"[{VERSION}] Initializing - Improved Black Frame Detection")
        self.replicate_client = None
    
    def detect_black_frame_precise(self, image_np):
        """정밀한 검은색 프레임 감지 - 50-80픽셀 두께 대응"""
        h, w = image_np.shape[:2]
        print(f"[{VERSION}] Detecting black frame in {w}x{h} image")
        
        # Step 1: 가장자리부터 검사 (안쪽으로 진행)
        edge_thickness = 0
        threshold = 40  # 검은색 판단 기준
        
        # 상단 가장자리 검사
        for i in range(min(200, h//2)):  # 최대 200픽셀까지 검사
            top_row = image_np[i, :]
            if np.mean(top_row) < threshold:
                edge_thickness = i + 1
            else:
                break
        
        if edge_thickness > 20:  # 20픽셀 이상이면 프레임으로 판단
            print(f"[{VERSION}] Black frame detected: {edge_thickness}px thick")
            
            # 더 정확한 경계 찾기
            # 하단, 좌측, 우측도 확인
            bottom_thickness = 0
            for i in range(min(200, h//2)):
                bottom_row = image_np[h-1-i, :]
                if np.mean(bottom_row) < threshold:
                    bottom_thickness = i + 1
                else:
                    break
            
            left_thickness = 0
            for i in range(min(200, w//2)):
                left_col = image_np[:, i]
                if np.mean(left_col) < threshold:
                    left_thickness = i + 1
                else:
                    break
            
            right_thickness = 0
            for i in range(min(200, w//2)):
                right_col = image_np[:, w-1-i]
                if np.mean(right_col) < threshold:
                    right_thickness = i + 1
                else:
                    break
            
            # 평균 두께 계산
            avg_thickness = (edge_thickness + bottom_thickness + left_thickness + right_thickness) / 4
            print(f"[{VERSION}] Frame thickness - T:{edge_thickness} B:{bottom_thickness} L:{left_thickness} R:{right_thickness}")
            
            # 마스크 생성
            mask = np.zeros((h, w), dtype=bool)
            mask[:edge_thickness, :] = True  # 상단
            mask[-bottom_thickness:, :] = True  # 하단
            mask[:, :left_thickness] = True  # 좌측
            mask[:, -right_thickness:] = True  # 우측
            
            return {
                'has_frame': True,
                'thickness': int(avg_thickness),
                'mask': mask,
                'bounds': (left_thickness, edge_thickness, w-right_thickness, h-bottom_thickness)
            }
        
        # Step 2: 중앙 검은색 박스 검사
        # 연속된 검은색 영역 찾기
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # 윤곽선 찾기
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w_box, h_box = cv2.boundingRect(contour)
            area = w_box * h_box
            
            # 전체 면적의 5% 이상인 검은색 영역
            if area > (w * h * 0.05):
                print(f"[{VERSION}] Black box detected at ({x},{y}) size {w_box}x{h_box}")
                
                mask = np.zeros((h, w), dtype=bool)
                mask[y:y+h_box, x:x+w_box] = True
                
                return {
                    'has_frame': True,
                    'thickness': 0,
                    'mask': mask,
                    'bounds': (x, y, x+w_box, y+h_box)
                }
        
        print(f"[{VERSION}] No black frame detected")
        return {'has_frame': False, 'thickness': 0, 'mask': None, 'bounds': (0, 0, w, h)}
    
    def remove_black_frame_replicate(self, image, frame_info):
        """Replicate API로 검은 프레임 제거"""
        if not REPLICATE_AVAILABLE or not frame_info['has_frame']:
            return image
        
        try:
            print(f"[{VERSION}] Removing black frame with Replicate")
            
            # 마스크 이미지 생성
            mask_np = frame_info['mask'].astype(np.uint8) * 255
            
            # 마스크 확장 (더 자연스러운 결과)
            kernel = np.ones((5, 5), np.uint8)
            mask_np = cv2.dilate(mask_np, kernel, iterations=2)
            
            mask_img = Image.fromarray(mask_np)
            
            # Base64 인코딩
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            
            mask_buffer = io.BytesIO()
            mask_img.save(mask_buffer, format='PNG')
            mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode('utf-8')
            
            # Replicate 실행
            if not self.replicate_client:
                self.replicate_client = replicate.Client(api_token=os.environ.get('REPLICATE_API_TOKEN'))
            
            output = self.replicate_client.run(
                "stability-ai/stable-diffusion-inpainting",
                input={
                    "image": f"data:image/png;base64,{img_base64}",
                    "mask": f"data:image/png;base64,{mask_base64}",
                    "prompt": "clean white background, professional product photography",
                    "num_inference_steps": 30
                }
            )
            
            if output and len(output) > 0:
                response = requests.get(output[0])
                return Image.open(io.BytesIO(response.content))
            
        except Exception as e:
            print(f"[{VERSION}] Replicate failed: {e}")
        
        # Fallback: 단순 크롭
        if frame_info['bounds']:
            x1, y1, x2, y2 = frame_info['bounds']
            return image.crop((x1, y1, x2, y2))
        
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
                          [-1,-1,-1]]) / 10
        sharpened = cv2.filter2D(thumb_np, -1, kernel)
        
        # 원본과 블렌딩
        result = cv2.addWeighted(thumb_np, 0.7, sharpened, 0.3, 0)
        
        print(f"[{VERSION}] Created {target_w}x{target_h} thumbnail with detail enhancement")
        
        return Image.fromarray(result.astype(np.uint8))

# 전역 인스턴스
processor_instance = None

def get_processor():
    global processor_instance
    if processor_instance is None:
        processor_instance = ThumbnailProcessorV16()
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
        
        # Google Script 호환을 위해 padding 유지
        base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
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
        
        # 1. 검은색 프레임 감지
        processor = get_processor()
        frame_info = processor.detect_black_frame_precise(image_np)
        
        # 2. 프레임 제거
        if frame_info['has_frame']:
            print(f"[{VERSION}] Removing black frame (thickness: {frame_info['thickness']}px)")
            image = processor.remove_black_frame_replicate(image, frame_info)
        
        # 3. 색감 보정 (Enhancement와 동일)
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
                "frame_thickness": frame_info['thickness'],
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
