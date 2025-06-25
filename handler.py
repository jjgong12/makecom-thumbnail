import runpod
import os
import sys
import json
import time
import base64
import numpy as np
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import cv2
import requests
import replicate
import logging
from typing import Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VERSION = "V69-ProperCrop"

# API Token - 환경변수 우선
REPLICATE_API_TOKEN = os.environ.get('REPLICATE_API_TOKEN', 'r8_6cksfxEmLxWlYxjW4K1FEbnZMEEmlQw2UeNNY')

def find_input_data(data):
    """재귀적으로 입력 데이터를 찾는 함수"""
    if isinstance(data, dict):
        if 'image' in data or 'url' in data or 'image_url' in data or 'imageUrl' in data or 'image_base64' in data or 'imageBase64' in data:
            return data
        
        if 'input' in data:
            return find_input_data(data['input'])
        
        for key in ['job', 'payload', 'data']:
            if key in data:
                result = find_input_data(data[key])
                if result:
                    return result
    
    return data

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
            
            content_type = response.headers.get('content-type', '')
            if 'image' not in content_type and not url.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                logger.warning(f"Warning: Non-image content type: {content_type}")
            
            return Image.open(BytesIO(response.content))
            
        except Exception as e:
            logger.error(f"Download attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                raise

def base64_to_image(base64_string):
    """Base64 문자열을 PIL Image로 변환"""
    # Remove data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    # padding 복원
    padding = 4 - len(base64_string) % 4
    if padding != 4:
        base64_string += '=' * padding
    
    img_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(img_data))

def apply_basic_enhancement(image):
    """Enhancement와 동일한 기본 보정 적용"""
    # RGB로 변환 (RGBA인 경우)
    if image.mode == 'RGBA':
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    
    # 1단계: 전체적으로 밝게
    brightness = ImageEnhance.Brightness(image)
    image = brightness.enhance(1.08)
    
    # 2단계: 약간의 대비 증가
    contrast = ImageEnhance.Contrast(image)
    image = contrast.enhance(1.05)
    
    # 3단계: 색상 살짝 증가
    color = ImageEnhance.Color(image)
    image = color.enhance(1.02)
    
    return image

def detect_ring_color(image):
    """반지 색상 감지 - Enhancement와 동일한 로직"""
    logger.info("Starting color detection...")
    
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    
    # 중앙 50% 영역 추출
    center_y, center_x = height // 2, width // 2
    crop_size = min(height, width) // 2
    
    y1 = max(0, center_y - crop_size // 2)
    y2 = min(height, center_y + crop_size // 2)
    x1 = max(0, center_x - crop_size // 2)
    x2 = min(width, center_x + crop_size // 2)
    
    center_region = img_array[y1:y2, x1:x2]
    
    # HSV로 변환
    hsv = cv2.cvtColor(center_region, cv2.COLOR_RGB2HSV)
    
    avg_saturation = np.mean(hsv[:, :, 1])
    avg_value = np.mean(hsv[:, :, 2])
    
    # RGB 분석
    rgb_center = center_region.astype(float)
    r_mean = np.mean(rgb_center[:, :, 0])
    g_mean = np.mean(rgb_center[:, :, 1])
    b_mean = np.mean(rgb_center[:, :, 2])
    
    # 정규화
    max_rgb = max(r_mean, g_mean, b_mean)
    if max_rgb > 0:
        r_norm = r_mean / max_rgb
        g_norm = g_mean / max_rgb
        b_norm = b_mean / max_rgb
    else:
        r_norm = g_norm = b_norm = 1.0
    
    logger.info(f"Color metrics - Sat: {avg_saturation:.1f}, Val: {avg_value:.1f}")
    logger.info(f"RGB normalized: R={r_norm:.2f}, G={g_norm:.2f}, B={b_norm:.2f}")
    
    # 색상 판단
    if avg_saturation < 25:
        if avg_value > 200:
            color = "무도금화이트"
        else:
            color = "화이트골드"
    elif r_norm > 0.95 and g_norm > 0.85 and g_norm < 0.95:
        if avg_saturation > 40:
            color = "로즈골드"
        else:
            color = "옐로우골드"
    elif abs(r_norm - g_norm) < 0.1 and abs(g_norm - b_norm) < 0.1:
        color = "화이트골드"
    else:
        warmth = (r_norm + g_norm) / 2 - b_norm
        if warmth > 0.1:
            color = "옐로우골드"
        else:
            color = "화이트골드"
    
    logger.info(f"Detected color: {color}")
    return color

def apply_color_specific_enhancement(image, color):
    """색상별 특별 보정 적용"""
    enhanced = image.copy()
    
    if color == '옐로우골드':
        # 옐로우골드: 따뜻한 톤 강화
        brightness = ImageEnhance.Brightness(enhanced)
        enhanced = brightness.enhance(1.1)
        
        # 노란색/금색 톤 강화
        img_array = np.array(enhanced)
        img_array[:, :, 0] = np.minimum(img_array[:, :, 0] * 1.05, 255).astype(np.uint8)  # R
        img_array[:, :, 1] = np.minimum(img_array[:, :, 1] * 1.03, 255).astype(np.uint8)  # G
        enhanced = Image.fromarray(img_array)
        
    elif color == '로즈골드':
        # 로즈골드: 핑크톤 강화
        brightness = ImageEnhance.Brightness(enhanced)
        enhanced = brightness.enhance(1.12)
        
        # 핑크/로즈 톤 강화
        img_array = np.array(enhanced)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = np.minimum(hsv[:, :, 1] * 1.1, 255).astype(np.uint8)
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
        # LAB 색공간으로 변환하여 순백색 만들기
        img_cv = cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(img_cv)
        
        # L채널(밝기) 대폭 증가
        l = cv2.multiply(l, 1.2)
        l = np.clip(l, 0, 255)
        
        # a, b 채널(색상) 감소 - 무채색으로
        a = cv2.multiply(a, 0.5)
        b = cv2.multiply(b, 0.5)
        
        img_cv = cv2.merge([l, a, b])
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_LAB2RGB)
        enhanced = Image.fromarray(img_rgb)
        
        # 추가 밝기 증가
        brightness = ImageEnhance.Brightness(enhanced)
        enhanced = brightness.enhance(1.3)
        
        # 채도 대폭 감소
        color_enhancer = ImageEnhance.Color(enhanced)
        enhanced = color_enhancer.enhance(0.3)
        
        # 블루 채널 살짝 증가 (노란빛 제거)
        img_array = np.array(enhanced)
        img_array[:, :, 2] = np.minimum(img_array[:, :, 2] * 1.1, 255).astype(np.uint8)
        enhanced = Image.fromarray(img_array)
    
    # 공통: 샤프니스 증가
    sharpness = ImageEnhance.Sharpness(enhanced)
    enhanced = sharpness.enhance(1.5)
    
    return enhanced

def create_thumbnail_with_crop(image, size=(1000, 1300)):
    """이미지를 1000x1300으로 크롭 - 30% 크롭으로 더 넓게"""
    logger.info(f"Creating thumbnail - original size: {image.size}")
    
    # 목표 비율
    target_ratio = 1000 / 1300  # 0.769
    img_width, img_height = image.size
    
    # 30% 크롭 (더 넓은 뷰)
    crop_percentage = 0.3
    
    # 크롭할 영역 계산
    crop_width = int(img_width * (1 - crop_percentage))
    crop_height = int(img_height * (1 - crop_percentage))
    
    # 비율 맞추기
    current_ratio = crop_width / crop_height
    
    if current_ratio > target_ratio:
        # 너무 넓음 - 높이를 늘림
        crop_height = int(crop_width / target_ratio)
    else:
        # 너무 높음 - 너비를 늘림
        crop_width = int(crop_height * target_ratio)
    
    # 중앙 정렬
    x_offset = (img_width - crop_width) // 2
    y_offset = (img_height - crop_height) // 2
    
    # 경계 체크
    x_offset = max(0, x_offset)
    y_offset = max(0, y_offset)
    crop_width = min(crop_width, img_width - x_offset)
    crop_height = min(crop_height, img_height - y_offset)
    
    # 크롭
    crop_box = (x_offset, y_offset, x_offset + crop_width, y_offset + crop_height)
    cropped = image.crop(crop_box)
    
    # 리사이즈
    thumbnail = cropped.resize(size, Image.Resampling.LANCZOS)
    
    logger.info(f"Thumbnail created: {thumbnail.size}")
    
    return thumbnail

def enhance_image_quality(image, replicate_token=None):
    """Replicate를 사용한 이미지 품질 향상"""
    try:
        if not replicate_token:
            replicate_token = REPLICATE_API_TOKEN
            logger.info("Using Replicate API token from environment variable")
        
        # Base64로 변환
        buffered = BytesIO()
        image.save(buffered, format="PNG", quality=100)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        img_uri = f"data:image/png;base64,{img_base64}"
        
        logger.info("Enhancing image quality with Real-ESRGAN...")
        
        # Real-ESRGAN 실행
        output = replicate.run(
            "nightmareai/real-esrgan:42fed1c4974146d4d2414e2be2c5277c7fcf05fcc3a73abf41610695738c1d7b",
            input={
                "image": img_uri,
                "scale": 2,
                "face_enhance": False
            }
        )
        
        # 결과 가져오기
        enhanced_url = output
        response = requests.get(enhanced_url)
        enhanced_image = Image.open(BytesIO(response.content))
        
        logger.info(f"Quality enhanced: {enhanced_image.size}")
        return enhanced_image
        
    except Exception as e:
        logger.error(f"Quality enhancement failed: {str(e)}")
        return image

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
    
    logger.info(f"Base64 length (no padding): {len(img_base64_no_padding)}")
    
    return img_base64_no_padding

def handler(event):
    """Thumbnail 핸들러 함수"""
    try:
        logger.info(f"[{VERSION}] Event received")
        logger.info(f"Event structure: {json.dumps(event, indent=2)[:500]}...")
        
        # 입력 데이터 찾기
        input_data = find_input_data(event)
        
        if not input_data:
            raise ValueError("No input data found")
        
        # 이미지 소스 확인
        image = None
        
        # Base64 입력 처리
        if isinstance(input_data, dict):
            if 'image_base64' in input_data or 'imageBase64' in input_data:
                base64_str = input_data.get('image_base64') or input_data.get('imageBase64')
                logger.info("Base64 image input detected")
                image = base64_to_image(base64_str)
            elif 'url' in input_data or 'image_url' in input_data or 'imageUrl' in input_data:
                image_url = input_data.get('url') or input_data.get('image_url') or input_data.get('imageUrl')
                logger.info(f"URL input detected: {image_url}")
                image = download_image_from_url(image_url)
        elif isinstance(input_data, str):
            if input_data.startswith('http'):
                logger.info(f"URL string input: {input_data}")
                image = download_image_from_url(input_data)
            else:
                logger.info("Base64 string input detected")
                image = base64_to_image(input_data)
        
        if not image:
            raise ValueError(f"Could not load image from input: {input_data}")
        
        logger.info(f"Image loaded: {image.size}")
        
        # 1. Enhancement 기본 보정 적용
        enhanced_image = apply_basic_enhancement(image)
        logger.info("Basic enhancement applied")
        
        # 2. 1000x1300으로 크롭 (30% 크롭으로 더 넓게)
        thumbnail = create_thumbnail_with_crop(enhanced_image)
        logger.info(f"Thumbnail cropped: {thumbnail.size}")
        
        # 3. 크롭된 이미지에서 색상 감지
        detected_color = detect_ring_color(thumbnail)
        logger.info(f"Detected ring color: {detected_color}")
        
        # 4. 색상별 특별 보정 적용
        thumbnail = apply_color_specific_enhancement(thumbnail, detected_color)
        
        # 5. Replicate API 토큰이 있으면 품질 향상
        replicate_token = input_data.get('replicate_api_token') if isinstance(input_data, dict) else None
        
        if replicate_token or REPLICATE_API_TOKEN:
            try:
                thumbnail = enhance_image_quality(thumbnail, replicate_token)
                # 품질 향상 후 다시 1000x1300으로 리사이즈
                if thumbnail.size != (1000, 1300):
                    thumbnail = thumbnail.resize((1000, 1300), Image.Resampling.LANCZOS)
            except Exception as e:
                logger.warning(f"Quality enhancement skipped: {str(e)}")
        
        # 6. 최종 보정
        # 중앙 포커스 효과
        width, height = thumbnail.size
        mask = Image.new('L', (width, height), 0)
        for y in range(height):
            for x in range(width):
                # 중앙으로부터의 거리 계산
                dx = abs(x - width/2) / (width/2)
                dy = abs(y - height/2) / (height/2)
                dist = max(dx, dy)
                # 부드러운 비네팅
                brightness = int(255 * (1 - dist * 0.2))
                mask.putpixel((x, y), brightness)
        
        # 비네팅 적용
        thumbnail = Image.composite(thumbnail, Image.new('RGB', thumbnail.size, (245, 245, 245)), mask)
        
        # base64 변환 (padding 제거)
        thumbnail_base64 = image_to_base64(thumbnail)
        
        # 중첩된 output 구조로 반환
        return {
            "output": {
                "thumbnail": thumbnail_base64,
                "size": list(thumbnail.size),
                "detected_color": detected_color,
                "format": "base64_no_padding",
                "version": VERSION
            }
        }
        
    except Exception as e:
        logger.error(f"Thumbnail error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "output": {
                "error": str(e),
                "status": "failed",
                "version": VERSION
            }
        }

# RunPod 핸들러 등록
runpod.serverless.start({"handler": handler})
