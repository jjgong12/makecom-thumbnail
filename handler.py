import runpod
import base64
import numpy as np
from PIL import Image, ImageEnhance
import cv2
import io
import os
import json
import traceback
import time
from typing import Dict, Any, Tuple, Optional, List

# Version info
VERSION = "v14-thumbnail"

# Import Replicate and requests only when available
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
    print(f"[{VERSION}] Requests not available")

class ThumbnailProcessorV14:
    """v14 Thumbnail Processor - Ultra Detection"""
    
    def __init__(self):
        print(f"[{VERSION}] Initializing - Ultra Detection & Make.com Fix")
        self.replicate_client = None
    
    def ultra_advanced_masking_detection(self, image_np):
        """초정밀 마스킹 감지 - 여러 단계 검증과 루프"""
        print(f"[{VERSION}] Starting ultra-advanced masking detection")
        
        h, w = image_np.shape[:2]
        detection_results = []
        
        # 1단계: 다양한 threshold로 검은색 영역 감지
        thresholds = [20, 25, 30, 35, 40, 45, 50, 60, 70, 80]
        for thresh in thresholds:
            # RGB 모두 threshold 이하인 픽셀
            black_mask = np.all(image_np < thresh, axis=2)
            black_ratio = np.sum(black_mask) / (h * w)
            
            if black_ratio > 0.01:  # 1% 이상이면 기록
                detection_results.append({
                    'threshold': thresh,
                    'black_ratio': black_ratio,
                    'mask': black_mask
                })
                print(f"[{VERSION}] Threshold {thresh}: {black_ratio:.2%} black pixels")
        
        # 2단계: 가장자리 검사 (각 방향별로 정밀 검사)
        edge_sizes = [50, 100, 150, 200, 250]
        edge_detections = {
            'top': [], 'bottom': [], 'left': [], 'right': []
        }
        
        for edge_size in edge_sizes:
            # 상단 가장자리
            top_region = image_np[:edge_size, :]
            for thresh in [30, 40, 50]:
                black_pixels = np.all(top_region < thresh, axis=2)
                ratio = np.sum(black_pixels) / black_pixels.size
                if ratio > 0.7:  # 70% 이상 검은색
                    edge_detections['top'].append((edge_size, ratio))
                    print(f"[{VERSION}] Top edge {edge_size}px: {ratio:.2%} black")
            
            # 하단 가장자리
            bottom_region = image_np[-edge_size:, :]
            for thresh in [30, 40, 50]:
                black_pixels = np.all(bottom_region < thresh, axis=2)
                ratio = np.sum(black_pixels) / black_pixels.size
                if ratio > 0.7:
                    edge_detections['bottom'].append((edge_size, ratio))
                    print(f"[{VERSION}] Bottom edge {edge_size}px: {ratio:.2%} black")
            
            # 좌측 가장자리
            left_region = image_np[:, :edge_size]
            for thresh in [30, 40, 50]:
                black_pixels = np.all(left_region < thresh, axis=2)
                ratio = np.sum(black_pixels) / black_pixels.size
                if ratio > 0.7:
                    edge_detections['left'].append((edge_size, ratio))
                    print(f"[{VERSION}] Left edge {edge_size}px: {ratio:.2%} black")
            
            # 우측 가장자리
            right_region = image_np[:, -edge_size:]
            for thresh in [30, 40, 50]:
                black_pixels = np.all(right_region < thresh, axis=2)
                ratio = np.sum(black_pixels) / black_pixels.size
                if ratio > 0.7:
                    edge_detections['right'].append((edge_size, ratio))
                    print(f"[{VERSION}] Right edge {edge_size}px: {ratio:.2%} black")
        
        # 3단계: 중앙 영역 집중 검사
        center_x, center_y = w // 2, h // 2
        scan_sizes = [50, 100, 150, 200, 250, 300, 400]  # 다양한 크기로 스캔
        
        for size in scan_sizes:
            x1 = max(0, center_x - size)
            y1 = max(0, center_y - size)
            x2 = min(w, center_x + size)
            y2 = min(h, center_y + size)
            
            center_region = image_np[y1:y2, x1:x2]
            
            # 중앙 영역의 검은색 비율 계산
            for thresh in [30, 40, 50]:
                black_pixels = np.all(center_region < thresh, axis=2)
                black_ratio = np.sum(black_pixels) / black_pixels.size
                
                if black_ratio > 0.3:  # 30% 이상이면 중앙 마스킹 의심
                    print(f"[{VERSION}] Center region {size}x{size}: {black_ratio:.2%} black (thresh={thresh})")
                    
                    # 사각형 형태인지 확인
                    if self.is_rectangular_mask(black_pixels):
                        return {
                            'has_masking': True,
                            'type': 'central_box',
                            'region': (x1, y1, x2, y2),
                            'confidence': black_ratio,
                            'mask': self.create_full_mask(image_np.shape[:2], (x1, y1, x2, y2))
                        }
        
        # 4단계: 연속된 검은색 영역 감지 (Connected Components)
        for result in detection_results[:3]:  # 가장 민감한 3개 threshold 검사
            mask = result['mask'].astype(np.uint8) * 255
            
            # 형태학적 연산으로 노이즈 제거
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Connected components 분석
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            
            for i in range(1, num_labels):  # 0은 배경
                area = stats[i, cv2.CC_STAT_AREA]
                if area > (h * w * 0.02):  # 전체 영역의 2% 이상
                    x, y, width, height = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                                         stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                    
                    # 사각형 형태인지 확인
                    aspect_ratio = width / height if height > 0 else 0
                    if 0.3 < aspect_ratio < 3.0:  # 너무 길거나 좁지 않은 형태
                        print(f"[{VERSION}] Found black region: {width}x{height} at ({x},{y})")
                        return {
                            'has_masking': True,
                            'type': 'detected_box',
                            'region': (x, y, x + width, y + height),
                            'confidence': area / (h * w),
                            'mask': self.create_full_mask(image_np.shape[:2], (x, y, x + width, y + height))
                        }
        
        # 5단계: 그라디언트 기반 검사
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Sobel 그라디언트
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # 급격한 변화가 있는 영역 찾기
        high_grad = grad_mag > np.percentile(grad_mag, 95)
        
        # 수평/수직 선 감지
        horizontal_lines = np.sum(high_grad, axis=1)
        vertical_lines = np.sum(high_grad, axis=0)
        
        # 강한 수평선 2개와 수직선 2개가 있으면 사각형 마스킹
        h_peaks = self.find_peaks(horizontal_lines, threshold=w * 0.3)
        v_peaks = self.find_peaks(vertical_lines, threshold=h * 0.3)
        
        if len(h_peaks) >= 2 and len(v_peaks) >= 2:
            print(f"[{VERSION}] Detected box edges via gradient")
            region = (v_peaks[0], h_peaks[0], v_peaks[-1], h_peaks[-1])
            return {
                'has_masking': True,
                'type': 'gradient_box',
                'region': region,
                'confidence': 0.8,
                'mask': self.create_full_mask(image_np.shape[:2], region)
            }
        
        # 6단계: 프레임 형태 종합 판단
        # 가장자리가 검은색인지 확인
        if len(edge_detections['top']) > 0 or len(edge_detections['bottom']) > 0 or \
           len(edge_detections['left']) > 0 or len(edge_detections['right']) > 0:
            
            # 가장 큰 검은색 영역 찾기
            max_top = max(edge_detections['top'], key=lambda x: x[0])[0] if edge_detections['top'] else 0
            max_bottom = max(edge_detections['bottom'], key=lambda x: x[0])[0] if edge_detections['bottom'] else 0
            max_left = max(edge_detections['left'], key=lambda x: x[0])[0] if edge_detections['left'] else 0
            max_right = max(edge_detections['right'], key=lambda x: x[0])[0] if edge_detections['right'] else 0
            
            if max_top > 0 or max_bottom > 0 or max_left > 0 or max_right > 0:
                print(f"[{VERSION}] Edge frame detected: T{max_top} B{max_bottom} L{max_left} R{max_right}")
                
                # 전체 마스크 생성
                full_mask = np.zeros((h, w), dtype=bool)
                if max_top > 0:
                    full_mask[:max_top, :] = True
                if max_bottom > 0:
                    full_mask[-max_bottom:, :] = True
                if max_left > 0:
                    full_mask[:, :max_left] = True
                if max_right > 0:
                    full_mask[:, -max_right:] = True
                
                return {
                    'has_masking': True,
                    'type': 'edge_frame',
                    'region': (max_left, max_top, w - max_right, h - max_bottom),
                    'confidence': np.sum(full_mask) / (h * w),
                    'mask': full_mask
                }
        
        print(f"[{VERSION}] No significant masking detected")
        return {'has_masking': False, 'type': None, 'region': None, 'confidence': 0, 'mask': None}
    
    def is_rectangular_mask(self, mask):
        """마스크가 사각형 형태인지 확인"""
        if mask.size == 0:
            return False
        
        # 행과 열의 합 계산
        row_sums = np.sum(mask, axis=1)
        col_sums = np.sum(mask, axis=0)
        
        # 연속된 영역 찾기
        row_mask = row_sums > mask.shape[1] * 0.5  # 50% 이상이 검은색
        col_mask = col_sums > mask.shape[0] * 0.5
        
        # 연속된 영역이 하나인지 확인
        row_changes = np.diff(np.concatenate(([False], row_mask, [False])).astype(int))
        col_changes = np.diff(np.concatenate(([False], col_mask, [False])).astype(int))
        
        row_regions = np.sum(row_changes == 1)
        col_regions = np.sum(col_changes == 1)
        
        return row_regions == 1 and col_regions == 1
    
    def find_peaks(self, arr, threshold):
        """배열에서 임계값을 넘는 피크 찾기"""
        peaks = []
        above_threshold = arr > threshold
        
        if not any(above_threshold):
            return peaks
        
        # 연속된 영역의 시작과 끝 찾기
        diff = np.diff(np.concatenate(([False], above_threshold, [False])).astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        for start, end in zip(starts, ends):
            peak_pos = start + np.argmax(arr[start:end])
            peaks.append(peak_pos)
        
        return peaks
    
    def create_full_mask(self, shape, region):
        """전체 크기의 마스크 생성"""
        mask = np.zeros(shape, dtype=bool)
        x1, y1, x2, y2 = region
        mask[y1:y2, x1:x2] = True
        return mask
    
    def remove_masking_with_replicate(self, image, masking_info):
        """Replicate API로 마스킹 제거"""
        if not REPLICATE_AVAILABLE or not REQUESTS_AVAILABLE or not masking_info['has_masking']:
            return image
        
        try:
            print(f"[{VERSION}] Removing masking with Replicate API")
            
            # 마스크가 이미 있으면 사용, 없으면 생성
            if masking_info.get('mask') is not None:
                mask = masking_info['mask'].astype(np.uint8) * 255
            else:
                # 마스크 생성
                mask = np.zeros((image.height, image.width), dtype=np.uint8)
                x1, y1, x2, y2 = masking_info['region']
                mask[y1:y2, x1:x2] = 255
            
            # 마스크 확장 (더 자연스러운 결과를 위해)
            kernel = np.ones((10, 10), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)
            
            # PIL Image로 변환
            mask_img = Image.fromarray(mask)
            
            # Base64 인코딩 (padding 제거!)
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8').rstrip('=')
            
            mask_buffer = io.BytesIO()
            mask_img.save(mask_buffer, format='PNG')
            mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode('utf-8').rstrip('=')
            
            # Replicate 실행
            if not self.replicate_client:
                self.replicate_client = replicate.Client(api_token=os.environ.get('REPLICATE_API_TOKEN'))
            
            output = self.replicate_client.run(
                "lucataco/flux-fill-pro",
                input={
                    "image": f"data:image/png;base64,{img_base64}",
                    "mask": f"data:image/png;base64,{mask_base64}",
                    "prompt": "professional product photography white seamless background gradient lighting",
                    "guidance_scale": 30,
                    "steps": 50,
                    "strength": 0.95
                }
            )
            
            if output:
                response = requests.get(output if isinstance(output, str) else output[0])
                return Image.open(io.BytesIO(response.content))
            
        except Exception as e:
            print(f"[{VERSION}] Replicate masking removal failed: {e}")
            traceback.print_exc()
        
        return image
    
    def apply_enhancement(self, image):
        """기본 이미지 향상 적용"""
        # 밝기 살짝 증가
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.1)
        
        # 대비 살짝 증가
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.05)
        
        # 선명도 증가
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.2)
        
        return image
    
    def create_perfect_thumbnail(self, image, target_size=(1000, 1300)):
        """완벽한 1000x1300 썸네일 생성"""
        img_np = np.array(image)
        h, w = img_np.shape[:2]
        target_w, target_h = target_size
        
        # 목표 비율
        target_ratio = target_w / target_h  # 0.769
        current_ratio = w / h
        
        if current_ratio > target_ratio:
            # 이미지가 더 넓음 - 좌우 잘라내기
            new_w = int(h * target_ratio)
            crop_x = (w - new_w) // 2
            cropped = img_np[:, crop_x:crop_x + new_w]
        else:
            # 이미지가 더 높음 - 상하 잘라내기
            new_h = int(w / target_ratio)
            crop_y = (h - new_h) // 2
            cropped = img_np[crop_y:crop_y + new_h, :]
        
        # 정확히 1000x1300으로 리사이즈
        thumbnail = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        
        print(f"[{VERSION}] Created perfect thumbnail: {target_w}x{target_h}")
        
        return Image.fromarray(thumbnail)

# 전역 인스턴스
processor_instance = None

def get_processor():
    """싱글톤 processor 인스턴스"""
    global processor_instance
    if processor_instance is None:
        processor_instance = ThumbnailProcessorV14()
    return processor_instance

def find_base64_in_dict(data, depth=0, max_depth=10):
    """중첩된 딕셔너리에서 base64 이미지 찾기"""
    if depth > max_depth:
        return None
    
    if isinstance(data, str) and len(data) > 100:
        return data
    
    if isinstance(data, dict):
        # 일반적인 키들 먼저 확인
        for key in ['image', 'base64', 'data', 'input', 'file']:
            if key in data and isinstance(data[key], str) and len(data[key]) > 100:
                return data[key]
        
        # 모든 값 확인
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
        # Data URL 형식 처리
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        
        # Padding 추가 시도
        padding = 4 - len(base64_str) % 4
        if padding != 4:
            base64_str += '=' * padding
        
        # 디코드
        img_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_data))
        
        # RGB로 변환
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        return img
        
    except Exception as e:
        print(f"[{VERSION}] Error decoding base64: {e}")
        raise

def encode_image_to_base64(image, format='PNG'):
    """이미지를 base64로 인코딩 (Make.com 호환 - padding 제거!)"""
    try:
        # numpy 배열인 경우 PIL Image로 변환
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # 버퍼에 저장
        buffer = io.BytesIO()
        image.save(buffer, format=format, quality=95 if format == 'JPEG' else None)
        buffer.seek(0)
        
        # Base64 인코딩 후 padding 제거!
        base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8').rstrip('=')
        
        return base64_str
        
    except Exception as e:
        print(f"[{VERSION}] Error encoding image: {e}")
        raise

def handler(job):
    """RunPod 썸네일 핸들러 함수"""
    try:
        start_time = time.time()
        job_input = job["input"]
        
        print(f"[{VERSION}] Thumbnail processing started")
        print(f"[{VERSION}] Input type: {type(job_input)}")
        
        # Base64 이미지 찾기
        base64_image = find_base64_in_dict(job_input)
        if not base64_image:
            print(f"[{VERSION}] No image data found in input")
            return {
                "output": {
                    "error": "No image data found",
                    "version": VERSION,
                    "success": False
                }
            }
        
        print(f"[{VERSION}] Found image data, length: {len(base64_image)}")
        
        # 이미지 디코드
        image = decode_base64_image(base64_image)
        print(f"[{VERSION}] Image decoded: {image.size}")
        
        # numpy 배열로 변환
        image_np = np.array(image)
        
        # 초정밀 마스킹 감지
        processor = get_processor()
        masking_info = processor.ultra_advanced_masking_detection(image_np)
        
        # 마스킹 제거
        if masking_info['has_masking']:
            print(f"[{VERSION}] Black masking detected! Type: {masking_info['type']}, Confidence: {masking_info['confidence']:.2%}")
            image = processor.remove_masking_with_replicate(image, masking_info)
        else:
            print(f"[{VERSION}] No black masking detected")
        
        # 기본 향상 적용
        image = processor.apply_enhancement(image)
        
        # 완벽한 썸네일 생성
        thumbnail = processor.create_perfect_thumbnail(image, (1000, 1300))
        
        # 결과 인코딩 (padding 제거!)
        thumbnail_base64 = encode_image_to_base64(thumbnail)
        
        print(f"[{VERSION}] Thumbnail encoded, length: {len(thumbnail_base64)}")
        
        # 처리 시간
        processing_time = time.time() - start_time
        print(f"[{VERSION}] Processing completed in {processing_time:.2f}s")
        
        # Make.com 호환 return 구조!
        return {
            "output": {
                "thumbnail": thumbnail_base64,
                "has_black_frame": masking_info['has_masking'],
                "frame_ratio": round(masking_info['confidence'], 3),
                "masking_type": masking_info['type'],
                "success": True,
                "version": VERSION,
                "processing_time": round(processing_time, 2),
                "original_size": list(image.size),
                "thumbnail_size": [1000, 1300]
            }
        }
        
    except Exception as e:
        error_msg = f"Error processing thumbnail: {str(e)}"
        print(f"[{VERSION}] {error_msg}")
        traceback.print_exc()
        
        return {
            "output": {
                "error": error_msg,
                "traceback": traceback.format_exc(),
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
    print(f"Requests Available: {REQUESTS_AVAILABLE}")
    print(f"OpenCV Available: {cv2 is not None}")
    print(f"NumPy Available: {np is not None}")
    print(f"Replicate Token Set: {bool(os.environ.get('REPLICATE_API_TOKEN'))}")
    print("="*70)
    
    runpod.serverless.start({"handler": handler})
