import runpod
import base64
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import cv2
import io
import os
import traceback
import time

# Version info
VERSION = "v22-thumbnail"

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

class ThumbnailProcessorV22:
    """v22 Thumbnail Processor - Aggressive Black Box Removal"""
    
    def __init__(self):
        print(f"[{VERSION}] Initializing - Aggressive Black Box Removal")
        self.replicate_client = None
    
    def detect_and_crop_black_box(self, image):
        """검은 박스를 감지하고 즉시 크롭 - 단순하고 확실하게"""
        img_np = np.array(image)
        h, w = img_np.shape[:2]
        print(f"[{VERSION}] Processing {w}x{h} image")
        
        # Grayscale 변환
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # 검은 영역 찾기 - 매우 관대한 threshold
        _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
        
        # 노이즈 제거
        kernel = np.ones((20, 20), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 흰색 영역(링이 있는 곳) 찾기
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 가장 큰 흰색 영역 찾기
            largest_white = max(contours, key=cv2.contourArea)
            x, y, cw, ch = cv2.boundingRect(largest_white)
            
            # 웨딩링이 있는 영역 확인
            white_area = cw * ch
            total_area = w * h
            
            if white_area < total_area * 0.8:  # 검은 프레임이 있다고 판단
                print(f"[{VERSION}] Black box detected! White area at ({x},{y}) size {cw}x{ch}")
                
                # 즉시 크롭 - 여백 추가
                margin = 30
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(w, x + cw + margin)
                y2 = min(h, y + ch + margin)
                
                cropped = image.crop((x1, y1, x2, y2))
                print(f"[{VERSION}] Cropped to remove black box: {cropped.size}")
                return cropped, True
        
        # Method 2: 엣지에서 안쪽으로 스캔
        # 각 방향에서 첫 번째 밝은 픽셀 찾기
        threshold = 100
        
        # Top edge
        top = 0
        for i in range(h//2):
            if np.max(gray[i, :]) > threshold:
                top = max(0, i - 20)
                break
        
        # Bottom edge
        bottom = h
        for i in range(h//2):
            if np.max(gray[h-1-i, :]) > threshold:
                bottom = min(h, h-i + 20)
                break
        
        # Left edge
        left = 0
        for i in range(w//2):
            if np.max(gray[:, i]) > threshold:
                left = max(0, i - 20)
                break
        
        # Right edge
        right = w
        for i in range(w//2):
            if np.max(gray[:, w-1-i]) > threshold:
                right = min(w, w-i + 20)
                break
        
        # 크롭이 필요한지 확인
        if top > 50 or (h - bottom) > 50 or left > 50 or (w - right) > 50:
            print(f"[{VERSION}] Edge black frame detected: T:{top} B:{bottom} L:{left} R:{right}")
            cropped = image.crop((left, top, right, bottom))
            return cropped, True
        
        print(f"[{VERSION}] No black box detected")
        return image, False
    
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
    
    def create_thumbnail_1000x1300(self, image):
        """정확히 1000x1300 썸네일 생성 - 웨딩링 중심"""
        target_size = (1000, 1300)
        
        # 이미 작은 이미지면 패딩 추가
        if image.size[0] < 1000 or image.size[1] < 1300:
            # 캔버스 생성
            canvas = Image.new('RGB', target_size, (245, 243, 240))
            
            # 이미지를 중앙에 배치
            paste_x = (1000 - image.size[0]) // 2
            paste_y = (1300 - image.size[1]) // 2
            canvas.paste(image, (paste_x, paste_y))
            
            image = canvas
        else:
            # 큰 이미지는 웨딩링 찾아서 크롭
            img_np = np.array(image)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
            # 엣지 검출로 웨딩링 위치 찾기
            edges = cv2.Canny(gray, 50, 150)
            
            # 엣지가 있는 영역의 바운딩 박스
            coords = np.column_stack(np.where(edges > 0))
            if len(coords) > 0:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                
                # 20% 패딩
                pad_x = int((x_max - x_min) * 0.2)
                pad_y = int((y_max - y_min) * 0.2)
                
                x_min = max(0, x_min - pad_x)
                y_min = max(0, y_min - pad_y)
                x_max = min(img_np.shape[1], x_max + pad_x)
                y_max = min(img_np.shape[0], y_max + pad_y)
                
                # 크롭
                cropped = image.crop((x_min, y_min, x_max, y_max))
            else:
                # 엣지 못찾으면 중앙 크롭
                cropped = image
            
            # 1000x1300 비율로 맞추기
            cropped.thumbnail((1000, 1300), Image.Resampling.LANCZOS)
            
            # 정확히 1000x1300 캔버스에 배치
            canvas = Image.new('RGB', target_size, (245, 243, 240))
            paste_x = (1000 - cropped.size[0]) // 2
            paste_y = (1300 - cropped.size[1]) // 2
            canvas.paste(cropped, (paste_x, paste_y))
            
            image = canvas
        
        # 최종 선명도 증가
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.5)
        
        print(f"[{VERSION}] Created exact 1000x1300 thumbnail")
        return image

def handler(job):
    """RunPod handler - Aggressive approach"""
    print(f"[{VERSION}] ====== Thumbnail Handler Started ======")
    
    try:
        job_input = job.get("input", {})
        
        # Find base64 image - same logic as enhancement
        base64_image = None
        
        if isinstance(job_input, dict):
            for key in ['image', 'base64', 'data', 'input', 'file', 'imageData']:
                if key in job_input:
                    value = job_input[key]
                    if isinstance(value, str) and len(value) > 100:
                        base64_image = value
                        print(f"[{VERSION}] Found image in key: {key}")
                        break
        
        if not base64_image and isinstance(job_input, dict):
            for key, value in job_input.items():
                if isinstance(value, dict):
                    for sub_key in ['image', 'base64', 'data']:
                        if sub_key in value and isinstance(value[sub_key], str) and len(value[sub_key]) > 100:
                            base64_image = value[sub_key]
                            print(f"[{VERSION}] Found image in nested: {key}.{sub_key}")
                            break
        
        if not base64_image and isinstance(job_input, str) and len(job_input) > 100:
            base64_image = job_input
        
        if not base64_image:
            return {
                "output": {
                    "thumbnail": None,
                    "error": "No image data found",
                    "success": False,
                    "version": VERSION
                }
            }
        
        # Process image
        if ',' in base64_image and base64_image.startswith('data:'):
            base64_image = base64_image.split(',')[1]
        
        base64_image = base64_image.strip()
        
        # Add padding for decoding
        padding = 4 - len(base64_image) % 4
        if padding != 4:
            base64_image += '=' * padding
        
        # Decode
        img_data = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(img_data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        print(f"[{VERSION}] Image decoded: {image.size}")
        
        # Create processor
        processor = ThumbnailProcessorV22()
        
        # 1. AGGRESSIVE BLACK BOX REMOVAL - 크롭 우선
        image, had_black_box = processor.detect_and_crop_black_box(image)
        
        # 2. Apply color enhancement
        image = processor.apply_simple_enhancement(image)
        
        # 3. Create exact 1000x1300 thumbnail
        thumbnail = processor.create_thumbnail_1000x1300(image)
        
        # Convert to base64
        buffer = io.BytesIO()
        thumbnail.save(buffer, format='PNG', quality=95)
        buffer.seek(0)
        
        thumbnail_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Remove padding for Make.com
        thumbnail_base64 = thumbnail_base64.rstrip('=')
        
        print(f"[{VERSION}] Thumbnail base64 length: {len(thumbnail_base64)}")
        
        # Return proper structure
        result = {
            "output": {
                "thumbnail": thumbnail_base64,
                "has_black_frame": had_black_box,
                "success": True,
                "version": VERSION,
                "thumbnail_size": [1000, 1300],
                "processing_method": "aggressive_crop"
            }
        }
        
        print(f"[{VERSION}] ====== Success - Returning Thumbnail ======")
        return result
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"[{VERSION}] ERROR: {error_msg}")
        traceback.print_exc()
        
        return {
            "output": {
                "thumbnail": None,
                "error": error_msg,
                "success": False,
                "version": VERSION
            }
        }

# RunPod serverless start
if __name__ == "__main__":
    print("="*70)
    print(f"Wedding Ring Thumbnail {VERSION}")
    print("Aggressive Black Box Removal - Crop First Approach")
    print("Output: Exact 1000x1300 thumbnail")
    print("Make.com path: {{4.data.output.output.thumbnail}}")
    print("="*70)
    
    runpod.serverless.start({"handler": handler})
