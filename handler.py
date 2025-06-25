import json
import runpod
import base64
import requests
import time
from io import BytesIO
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import replicate
import os
import cv2

# Replicate API 설정
REPLICATE_API_TOKEN = "r8_8pH3riHZWKr6UwhUjVqHoNDrWqpOdek2nwdRa"
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN
print(f"Replicate API Token 설정됨: {REPLICATE_API_TOKEN[:10]}...")

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

def remove_background_with_replicate(image):
    """Replicate API를 사용하여 배경 제거"""
    # PIL Image를 base64로 변환
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    # Replicate 모델 실행
    model = replicate.models.get("cjwbw/rembg")
    version = model.versions.get("fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003")
    
    input_data = {
        "image": f"data:image/png;base64,{img_base64}"
    }
    
    print("Replicate API 호출 중...")
    output = version.predict(**input_data)
    
    # 결과 다운로드
    if isinstance(output, str) and output.startswith('http'):
        response = requests.get(output)
        return Image.open(BytesIO(response.content))
    elif isinstance(output, str) and 'base64,' in output:
        img_data = output.split('base64,')[1]
        return Image.open(BytesIO(base64.b64decode(img_data)))
    else:
        # 이미 PIL Image인 경우
        return output

def detect_ring_color(image):
    """반지 색상 감지 - 무도금화이트 우선 감지"""
    img_array = np.array(image)
    
    # RGB인 경우만 처리 (RGBA는 RGB로 변환)
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        # 흰색 배경에 합성
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

def create_thumbnail_with_color(image, detected_color, size=(1000, 1300)):
    """1000x1300 썸네일 생성 with 색상별 보정"""
    print(f"썸네일 생성 시작 - 색상: {detected_color}")
    
    # 배경이 제거된 이미지 처리
    img_array = np.array(image)
    
    # 알파 채널이 있는지 확인
    if img_array.shape[2] == 4:
        # 알파 채널을 사용하여 객체의 경계 찾기
        alpha = img_array[:, :, 3]
        coords = cv2.findNonZero(alpha)
        x, y, w, h = cv2.boundingRect(coords)
        
        # 객체 주변에 여백 추가 (15%)
        padding = int(max(w, h) * 0.15)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img_array.shape[1] - x, w + 2 * padding)
        h = min(img_array.shape[0] - y, h + 2 * padding)
        
        # 1000:1300 비율로 맞추기
        target_ratio = 1000 / 1300  # 0.769
        current_ratio = w / h
        
        if current_ratio > target_ratio:
            # 너무 넓음 - 높이를 늘려야 함
            new_h = int(w / target_ratio)
            diff = new_h - h
            y = max(0, y - diff // 2)
            h = new_h
        else:
            # 너무 높음 - 너비를 늘려야 함
            new_w = int(h * target_ratio)
            diff = new_w - w
            x = max(0, x - diff // 2)
            w = new_w
        
        # 경계 체크
        x = max(0, x)
        y = max(0, y)
        w = min(img_array.shape[1] - x, w)
        h = min(img_array.shape[0] - y, h)
        
        # 크롭
        cropped = img_array[y:y+h, x:x+w]
        cropped_img = Image.fromarray(cropped)
    else:
        cropped_img = image
    
    # 리사이즈
    thumbnail = cropped_img.resize(size, Image.Resampling.LANCZOS)
    
    # 흰색 배경 추가
    if thumbnail.mode == 'RGBA':
        background = Image.new('RGB', size, (255, 255, 255))
        background.paste(thumbnail, (0, 0), thumbnail)
        thumbnail = background
    
    # 색상별 디테일 보정 적용
    thumbnail = apply_color_specific_enhancement(thumbnail, detected_color)
    
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
        
        # 색상 감지 (배경 제거 전에 원본에서)
        detected_color = detect_ring_color(image)
        print(f"감지된 반지 색상: {detected_color}")
        
        # 배경 제거
        no_bg_image = remove_background_with_replicate(image)
        print("배경 제거 완료")
        
        # 썸네일 생성 (색상 정보 전달)
        thumbnail = create_thumbnail_with_color(no_bg_image, detected_color)
        print(f"썸네일 생성 완료: {thumbnail.size}")
        
        # base64 변환 (padding 제거)
        thumbnail_base64 = image_to_base64(thumbnail)
        
        # 중첩된 output 구조로 반환
        return {
            "output": {
                "thumbnail": thumbnail_base64,
                "size": list(thumbnail.size),
                "detected_color": detected_color,
                "format": "base64_no_padding"
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
