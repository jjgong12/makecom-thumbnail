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
    
    # 재귀적 탐색
    def recursive_search(obj, target_keys=['input', 'url', 'image_url', 'imageUrl']):
        if isinstance(obj, dict):
            for key in target_keys:
                if key in obj:
                    return obj[key]
            
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

def create_thumbnail(image, size=(500, 500)):
    """정사각형 썸네일 생성"""
    # 배경이 제거된 이미지를 정사각형으로 만들기
    img_array = np.array(image)
    
    # 알파 채널이 있는지 확인
    if img_array.shape[2] == 4:
        # 알파 채널을 사용하여 객체의 경계 찾기
        alpha = img_array[:, :, 3]
        coords = cv2.findNonZero(alpha)
        x, y, w, h = cv2.boundingRect(coords)
        
        # 객체 주변에 여백 추가 (10%)
        padding = int(max(w, h) * 0.1)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img_array.shape[1] - x, w + 2 * padding)
        h = min(img_array.shape[0] - y, h + 2 * padding)
        
        # 정사각형으로 만들기
        if w > h:
            diff = w - h
            y = max(0, y - diff // 2)
            h = w
        else:
            diff = h - w
            x = max(0, x - diff // 2)
            w = h
        
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
        
        # URL 추출
        image_url = None
        if isinstance(input_data, dict):
            image_url = input_data.get('url') or input_data.get('image_url') or input_data.get('imageUrl')
        elif isinstance(input_data, str):
            image_url = input_data
        
        if not image_url:
            raise ValueError(f"이미지 URL을 찾을 수 없습니다. 입력: {input_data}")
        
        print(f"이미지 URL: {image_url}")
        
        # 이미지 다운로드
        image = download_image_from_url(image_url)
        print(f"이미지 다운로드 완료: {image.size}")
        
        # 배경 제거
        no_bg_image = remove_background_with_replicate(image)
        print("배경 제거 완료")
        
        # 썸네일 생성
        thumbnail = create_thumbnail(no_bg_image)
        print(f"썸네일 생성 완료: {thumbnail.size}")
        
        # base64 변환 (padding 제거)
        thumbnail_base64 = image_to_base64(thumbnail)
        
        # 중첩된 output 구조로 반환
        return {
            "output": {
                "thumbnail": thumbnail_base64,
                "size": thumbnail.size,
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
