import json
import runpod
import base64
import requests
import time
from io import BytesIO
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import cv2

def find_input_data(data):
    """재귀적으로 모든 가능한 경로에서 입력 데이터 찾기"""
    
    # 전체 구조 로깅 (디버깅용)
    print(f"전체 입력 데이터 구조: {json.dumps(data, indent=2)[:1000]}")
    
    # 직접 접근 시도
    if isinstance(data, dict):
        # 최상위 레벨 체크
        if 'input' in data:
            return data['input']
        
        # 일반적인 RunPod 구조들
        common_paths = [
            ['job', 'input'],
            ['data', 'input'],
            ['payload', 'input'],
            ['body', 'input'],
            ['request', 'input']
        ]
        
        for path in common_paths:
            current = data
            for key in path:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    break
            else:
                return current
    
    # 재귀적 탐색 - image_base64도 찾기
    def recursive_search(obj, target_keys=['input', 'url', 'image_url', 'imageUrl', 'image_base64', 'imageBase64']):
        if isinstance(obj, dict):
            for key in target_keys:
                if key in obj:
                    return obj[key] if key == 'input' else {key: obj[key]}
            
            for value in obj.values():
                result = recursive_search(value, target_keys)
                if result:
                    return result
        elif isinstance(obj, list):
            for item in obj:
                result = recursive_search(item, target_keys)
                if result:
                    return result
        
        return None
    
    result = recursive_search(data)
    print(f"재귀 탐색 결과: {result}")
    return result

def download_image_from_url(url):
    """URL에서 이미지를 다운로드하여 PIL Image 객체로 반환"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8'
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # 컨텐츠 타입 확인
            content_type = response.headers.get('content-type', '')
            if 'image' not in content_type and not url.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                print(f"경고: 이미지가 아닌 컨텐츠 타입: {content_type}")
            
            return Image.open(BytesIO(response.content))
            
        except Exception as e:
            print(f"다운로드 시도 {attempt + 1}/{max_retries} 실패: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                raise

def base64_to_image(base64_string):
    """Base64 문자열을 PIL Image로 변환"""
    # padding 복원
    padding = 4 - len(base64_string) % 4
    if padding != 4:
        base64_string += '=' * padding
    
    # data URL 형식 처리
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    img_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(img_data))

def apply_basic_enhancement(image):
    """Enhancement와 동일한 기본 보정 적용 (V58 버전)"""
    # RGB로 변환 (RGBA인 경우)
    if image.mode == 'RGBA':
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    
    # 1단계: 전체적으로 밝게 (35%)
    brightness = ImageEnhance.Brightness(image)
    image = brightness.enhance(1.35)
    
    # 2단계: 대비 약간 감소
    contrast = ImageEnhance.Contrast(image)
    image = contrast.enhance(0.9)
    
    # 3단계: 채도 감소
    color = ImageEnhance.Color(image)
    image = color.enhance(0.7)
    
    # 4단계: 감마 보정
    img_array = np.array(image)
    
    for i in range(3):
        channel = img_array[:, :, i].astype(np.float32) / 255.0
        channel = np.power(channel, 0.7)  # 감마 0.7
        channel = np.where(channel > 0.6, 
                          channel + (1 - channel) * 0.15,
                          channel * 1.05)
        channel = np.clip(channel, 0, 1)
        img_array[:, :, i] = (channel * 255).astype(np.uint8)
    
    # 5단계: 화이트 오버레이 제거 (0%)
    # 화이트 오버레이 스킵
    
    enhanced_image = Image.fromarray(img_array)
    
    # 6단계: 추가 밝기 (5%)
    brightness2 = ImageEnhance.Brightness(enhanced_image)
    enhanced_image = brightness2.enhance(1.05)
    
    # 7단계: 샤프니스
    sharpness = ImageEnhance.Sharpness(enhanced_image)
    enhanced_image = sharpness.enhance(1.1)
    
    return enhanced_image

def detect_ring_color(image):
    """반지 색상 감지 - 무도금화이트 우선 감지"""
    img_array = np.array(image)
    
    # RGB인 경우만 처리
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        rgb_image = Image.new('RGB', image.size, (255, 255, 255))
        rgb_image.paste(image, mask=image.split()[3])
        img_array = np.array(rgb_image)
    
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    # 이미지 중앙 부분만 분석 (반지가 있을 가능성이 높은 부분)
    h, w = hsv.shape[:2]
    center_region = hsv[h//4:3*h//4, w//4:3*w//4]
    
    # 각 색상별 픽셀 수 계산
    color_pixels = {}
    
    # 1. 무도금화이트 우선 체크 (넓은 범위)
    white_mask = cv2.inRange(center_region, 
                            np.array([0, 0, 180]),      # 매우 밝은 영역
                            np.array([180, 30, 255]))    # 채도 낮음
    color_pixels['white'] = cv2.countNonZero(white_mask)
    
    # 2. 옐로우골드
    yellow_mask = cv2.inRange(center_region,
                             np.array([20, 50, 100]),
                             np.array([30, 255, 255]))
    color_pixels['yellow'] = cv2.countNonZero(yellow_mask)
    
    # 3. 로즈골드
    rose_mask = cv2.inRange(center_region,
                           np.array([0, 30, 100]),
                           np.array([15, 150, 255]))
    color_pixels['rose'] = cv2.countNonZero(rose_mask)
    
    # 4. 화이트골드
    white_gold_mask = cv2.inRange(center_region,
                                 np.array([0, 0, 150]),
                                 np.array([180, 40, 230]))
    color_pixels['white_gold'] = cv2.countNonZero(white_gold_mask)
    
    # 무도금화이트가 전체의 70% 이상이면 무도금화이트로 판정
    total_pixels = center_region.shape[0] * center_region.shape[1]
    if color_pixels['white'] > total_pixels * 0.7:
        return '무도금화이트'
    
    # 그 외의 경우 가장 많은 픽셀 수를 가진 색상 선택
    detected_color = max(color_pixels.items(), key=lambda x: x[1])[0]
    
    color_map = {
        'yellow': '옐로우골드',
        'rose': '로즈골드',
        'white_gold': '화이트골드',
        'white': '무도금화이트'
    }
    
    return color_map.get(detected_color, '무도금화이트')

def apply_color_specific_enhancement(image, color):
    """색상별 특화 보정 적용"""
    enhanced = image.copy()
    
    if color == '옐로우골드':
        # 옐로우골드: 따뜻한 톤 강화, 광택 증가
        brightness = ImageEnhance.Brightness(enhanced)
        enhanced = brightness.enhance(1.2)
        
        # 노란색 채널 강화
        img_array = np.array(enhanced)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        # 노란색 영역의 채도 증가
        yellow_mask = cv2.inRange(h, 20, 30)
        s = np.where(yellow_mask > 0, np.minimum(s * 1.3, 255), s).astype(np.uint8)
        
        hsv = cv2.merge([h, s, v])
        enhanced = Image.fromarray(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
        
    elif color == '로즈골드':
        # 로즈골드: 핑크톤 강화, 부드러운 광택
        img_array = np.array(enhanced)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        # 붉은 톤 영역 강화
        rose_mask = cv2.inRange(h, 0, 15)
        s = np.where(rose_mask > 0, np.minimum(s * 1.2, 255), s).astype(np.uint8)
        v = np.where(rose_mask > 0, np.minimum(v * 1.1, 255), v).astype(np.uint8)
        
        hsv = cv2.merge([h, s, v])
        enhanced = Image.fromarray(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
        
    elif color == '화이트골드':
        # 화이트골드: 차가운 톤, 메탈릭 광택
        brightness = ImageEnhance.Brightness(enhanced)
        enhanced = brightness.enhance(1.15)
        
        contrast = ImageEnhance.Contrast(enhanced)
        enhanced = contrast.enhance(1.1)
        
        # 약간의 블루 톤 추가
        img_array = np.array(enhanced)
        img_array[:, :, 2] = np.minimum(img_array[:, :, 2] * 1.05, 255).astype(np.uint8)
        enhanced = Image.fromarray(img_array)
        
    else:  # 무도금화이트
        # 무도금화이트: 순백색 강조, 밝기 최대화
        brightness = ImageEnhance.Brightness(enhanced)
        enhanced = brightness.enhance(1.25)
        
        # 채도 감소로 더 하얗게
        color_enhancer = ImageEnhance.Color(enhanced)
        enhanced = color_enhancer.enhance(0.7)
    
    # 공통: 샤프니스 증가
    sharpness = ImageEnhance.Sharpness(enhanced)
    enhanced = sharpness.enhance(1.5)
    
    return enhanced

def create_thumbnail_with_crop(image, size=(1000, 1300)):
    """이미지를 1000x1300으로 크롭 (중앙 기준)"""
    print(f"썸네일 크롭 시작 - 원본 크기: {image.size}")
    
    # 목표 비율
    target_ratio = 1000 / 1300  # 0.769
    img_width, img_height = image.size
    current_ratio = img_width / img_height
    
    # 크롭할 영역 계산
    if current_ratio > target_ratio:
        # 이미지가 너무 넓음 - 좌우를 자름
        new_width = int(img_height * target_ratio)
        x_offset = (img_width - new_width) // 2
        crop_box = (x_offset, 0, x_offset + new_width, img_height)
    else:
        # 이미지가 너무 높음 - 상하를 자름
        new_height = int(img_width / target_ratio)
        y_offset = (img_height - new_height) // 2
        crop_box = (0, y_offset, img_width, y_offset + new_height)
    
    # 크롭
    cropped = image.crop(crop_box)
    
    # 리사이즈
    thumbnail = cropped.resize(size, Image.Resampling.LANCZOS)
    
    return thumbnail

def image_to_base64(image, format='JPEG'):
    """PIL Image를 base64 문자열로 변환"""
    buffered = BytesIO()
    if format == 'JPEG' and image.mode == 'RGBA':
        rgb_image = Image.new('RGB', image.size, (255, 255, 255))
        rgb_image.paste(image, mask=image.split()[3])
        image = rgb_image
    
    image.save(buffered, format=format, quality=95)
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    # Make.com을 위해 padding 제거
    img_base64_no_padding = img_base64.rstrip('=')
    
    print(f"Base64 길이 (padding 제거): {len(img_base64_no_padding)}")
    
    return img_base64_no_padding

def handler(event):
    """Thumbnail 핸들러 함수"""
    try:
        print("Thumbnail 이벤트 수신:", json.dumps(event, indent=2)[:500])
        
        # 입력 데이터 찾기
        input_data = find_input_data(event)
        
        if not input_data:
            raise ValueError("입력 데이터를 찾을 수 없습니다")
        
        # 이미지 소스 확인 (URL 또는 Base64)
        image = None
        
        # Base64 입력 처리
        if isinstance(input_data, dict):
            if 'image_base64' in input_data or 'imageBase64' in input_data:
                base64_str = input_data.get('image_base64') or input_data.get('imageBase64')
                print("Base64 이미지 입력 감지")
                image = base64_to_image(base64_str)
            elif 'url' in input_data or 'image_url' in input_data or 'imageUrl' in input_data:
                image_url = input_data.get('url') or input_data.get('image_url') or input_data.get('imageUrl')
                print(f"URL 입력 감지: {image_url}")
                image = download_image_from_url(image_url)
        elif isinstance(input_data, str):
            # 문자열인 경우 URL로 가정
            if input_data.startswith('http'):
                print(f"URL 문자열 입력: {input_data}")
                image = download_image_from_url(input_data)
            else:
                # Base64 문자열로 가정
                print("Base64 문자열 입력 감지")
                image = base64_to_image(input_data)
        
        if not image:
            raise ValueError(f"이미지를 로드할 수 없습니다. 입력: {input_data}")
        
        print(f"이미지 로드 완료: {image.size}")
        
        # 1. Enhancement 기본 보정 적용 (V58 버전)
        enhanced_image = apply_basic_enhancement(image)
        print("기본 보정 적용 완료 (V58)")
        
        # 2. 1000x1300으로 크롭
        thumbnail = create_thumbnail_with_crop(enhanced_image)
        print(f"썸네일 크롭 완료: {thumbnail.size}")
        
        # 3. 크롭된 이미지에서 색상 감지
        detected_color = detect_ring_color(thumbnail)
        print(f"감지된 반지 색상: {detected_color}")
        
        # 4. 색상별 추가 보정 적용
        thumbnail = apply_color_specific_enhancement(thumbnail, detected_color)
        print("색상별 보정 적용 완료")
        
        # base64 변환 (padding 제거)
        thumbnail_base64 = image_to_base64(thumbnail)
        
        # 중첩된 output 구조로 반환
        return {
            "output": {
                "thumbnail": thumbnail_base64,
                "size": list(thumbnail.size),
                "detected_color": detected_color,
                "format": "base64_no_padding",
                "process": "enhancement_crop_detect_enhance_v58"
            }
        }
        
    except Exception as e:
        print(f"Thumbnail 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "output": {
                "error": str(e),
                "status": "failed"
            }
        }

# RunPod 핸들러 등록
runpod.serverless.start({"handler": handler})
