import runpod
import base64
import requests
from io import BytesIO
from PIL import Image, ImageEnhance
import numpy as np
import replicate
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Replicate API 키 설정
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")
if REPLICATE_API_TOKEN:
    os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

def detect_black_frame(img_array, threshold=30):
    """
    검은색 프레임 영역 정확히 감지
    """
    # RGB 모두 threshold 이하인 픽셀을 검은색으로 판단
    black_mask = np.all(img_array < threshold, axis=2)
    
    # 검은색이 아닌 영역의 경계 찾기
    non_black = ~black_mask
    rows = np.any(non_black, axis=1)
    cols = np.any(non_black, axis=0)
    
    if not rows.any() or not cols.any():
        return None
    
    # 실제 콘텐츠가 있는 영역
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    logger.info(f"Content area: ({cmin}, {rmin}) to ({cmax}, {rmax})")
    
    # 프레임 영역 마스크 생성
    frame_mask = np.ones_like(black_mask, dtype=bool)
    frame_mask[rmin:rmax+1, cmin:cmax+1] = False
    
    # 실제로 검은색인 부분만 마스킹
    final_mask = black_mask & frame_mask
    
    return final_mask

def remove_black_frame_with_inpainting(img, mask):
    """
    Replicate API를 사용하여 검은색 프레임 인페인팅
    """
    try:
        # 마스크를 이미지로 변환
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        
        # 이미지와 마스크를 base64로 인코딩
        img_buffer = BytesIO()
        img.save(img_buffer, format="PNG")
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        mask_buffer = BytesIO()
        mask_img.save(mask_buffer, format="PNG")
        mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode('utf-8')
        
        # Replicate 모델 실행
        model = replicate.models.get("stability-ai/stable-diffusion-inpainting")
        version = model.versions.get("latest")
        
        output = replicate.run(
            f"{model.owner}/{model.name}:{version.id}",
            input={
                "image": f"data:image/png;base64,{img_base64}",
                "mask": f"data:image/png;base64,{mask_base64}",
                "prompt": "clean professional product photography background, smooth gradient, studio lighting",
                "negative_prompt": "black frame, border, dark edges",
                "num_inference_steps": 30,
                "guidance_scale": 7.5
            }
        )
        
        # 결과 다운로드
        if output and len(output) > 0:
            response = requests.get(output[0])
            result_img = Image.open(BytesIO(response.content))
            return result_img
        else:
            logger.error("No output from Replicate")
            return img
            
    except Exception as e:
        logger.error(f"Inpainting error: {str(e)}")
        return img

def simple_frame_removal(img, frame_mask):
    """
    간단한 프레임 제거 (인페인팅 실패 시 폴백)
    """
    img_array = np.array(img)
    
    # 프레임 영역을 주변 색상으로 채우기
    if frame_mask.any():
        # 콘텐츠 영역의 평균 배경색 계산
        content_mask = ~frame_mask
        
        # 가장자리 픽셀들의 평균색 계산
        edge_pixels = []
        h, w = frame_mask.shape
        
        # 상단, 하단, 좌측, 우측 가장자리 픽셀 수집
        for i in range(w):
            if content_mask[0, i]:
                edge_pixels.append(img_array[0, i])
            if content_mask[h-1, i]:
                edge_pixels.append(img_array[h-1, i])
        
        for i in range(h):
            if content_mask[i, 0]:
                edge_pixels.append(img_array[i, 0])
            if content_mask[i, w-1]:
                edge_pixels.append(img_array[i, w-1])
        
        if edge_pixels:
            avg_color = np.mean(edge_pixels, axis=0).astype(np.uint8)
        else:
            avg_color = np.array([240, 240, 240])  # 기본 밝은 회색
        
        # 프레임 영역을 평균색으로 채우기
        img_array[frame_mask] = avg_color
    
    return Image.fromarray(img_array)

def apply_enhancement(img):
    """
    enhance_handler와 동일한 색감 보정 적용
    """
    # 1. 아주 약간의 밝기 증가 (1.05 = 5% 증가)
    brightness_enhancer = ImageEnhance.Brightness(img)
    img = brightness_enhancer.enhance(1.05)
    
    # 2. 아주 약간의 채도 감소로 더 깨끗한 느낌 (0.95 = 5% 감소)
    color_enhancer = ImageEnhance.Color(img)
    img = color_enhancer.enhance(0.95)
    
    # 3. 배경만 살짝 밝게 (선택적)
    img_array = np.array(img)
    mask = np.all(img_array > 200, axis=-1)
    if mask.any():
        for c in range(3):
            img_array[mask, c] = np.minimum(255, img_array[mask, c] * 1.05).astype(np.uint8)
    
    return Image.fromarray(img_array)

def crop_to_ring_area(img, padding_ratio=0.05):
    """
    웨딩링 영역을 감지하고 90% 꽉 차게 크롭
    """
    img_array = np.array(img)
    gray = np.mean(img_array, axis=2)
    
    # 배경이 아닌 영역 찾기 (웨딩링 영역)
    non_bg = gray < 200  # 배경보다 어두운 부분
    rows = np.any(non_bg, axis=1)
    cols = np.any(non_bg, axis=0)
    
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # 10% 여백을 위해 영역 확장
        height = rmax - rmin
        width = cmax - cmin
        padding_h = int(height * padding_ratio)  # 상하 5%씩
        padding_w = int(width * padding_ratio)   # 좌우 5%씩
        
        # 경계 조정
        rmin = max(0, rmin - padding_h)
        rmax = min(img_array.shape[0], rmax + padding_h)
        cmin = max(0, cmin - padding_w)
        cmax = min(img_array.shape[1], cmax + padding_w)
        
        # 크롭
        img = img.crop((cmin, rmin, cmax, rmax))
        logger.info(f"Cropped to ring area: {cmax-cmin}x{rmax-rmin}")
    
    return img

def resize_to_thumbnail(img, target_width=1000, target_height=1300):
    """
    1000x1300 썸네일로 리사이즈 (비율 유지)
    """
    # 비율 계산
    img_width, img_height = img.size
    width_ratio = target_width / img_width
    height_ratio = target_height / img_height
    
    # 더 작은 비율 사용 (전체가 들어가도록)
    ratio = min(width_ratio, height_ratio)
    new_width = int(img_width * ratio)
    new_height = int(img_height * ratio)
    
    # 리사이즈
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # 1000x1300 캔버스 중앙에 배치
    final_img = Image.new('RGB', (target_width, target_height), (250, 250, 250))
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    final_img.paste(img, (x_offset, y_offset))
    
    logger.info(f"Final thumbnail created: {target_width}x{target_height} with rings at 90%")
    return final_img

def find_image_data(job_input):
    """
    다양한 키에서 이미지 데이터 찾기
    """
    logger.info(f"Input keys: {list(job_input.keys())}")
    
    # 가능한 키들 체크
    possible_keys = ['image', 'image_base64', 'enhanced_image', 'base64', 'img', 'data', 'imageData', 'image_url']
    
    # 1. 직접 키 체크
    for key in possible_keys:
        if key in job_input and job_input[key]:
            logger.info(f"Found image in key: {key}")
            return job_input[key]
    
    # 2. 중첩된 구조 체크
    nested_paths = [
        ['input', 'image'],
        ['data', 'image'],
        ['body', 'image'],
        ['payload', 'image'],
        ['input', 'enhanced_image'],
        ['data', 'enhanced_image']
    ]
    
    for path in nested_paths:
        current = job_input
        try:
            for key in path:
                current = current.get(key, {})
            if current and isinstance(current, str):
                logger.info(f"Found image in nested path: {'.'.join(path)}")
                return current
        except:
            continue
    
    # 3. 모든 키 순회하며 base64 패턴 찾기
    for key, value in job_input.items():
        if isinstance(value, str) and len(value) > 1000:  # base64는 보통 길다
            if value.startswith('data:image') or looks_like_base64(value):
                logger.info(f"Found base64-like string in key: {key}")
                return value
        elif isinstance(value, dict):
            # 재귀적으로 찾기
            result = find_image_data(value)
            if result:
                return result
    
    return None

def looks_like_base64(s):
    """Base64 패턴인지 확인"""
    import re
    # Base64 패턴 체크
    base64_pattern = re.compile(r'^[A-Za-z0-9+/]*={0,2}
        
        # 이미지 열기
        img = Image.open(BytesIO(image_data))
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        
        # 원본 크기 저장
        original_size = img.size
        
        # 검은색 프레임 감지
        img_array = np.array(img)
        frame_mask = detect_black_frame(img_array)
        
        if frame_mask is not None and frame_mask.any():
            logger.info("Black frame detected, removing...")
            
            # Replicate API로 인페인팅 시도
            if REPLICATE_API_TOKEN:
                result_img = remove_black_frame_with_inpainting(img, frame_mask)
            else:
                logger.warning("Replicate API token not found, using simple removal")
                result_img = simple_frame_removal(img, frame_mask)
        else:
            logger.info("No black frame detected")
            result_img = img
        
        # enhance와 동일한 색감 보정 적용
        result_img = apply_enhancement(result_img)
        
        # 웨딩링 영역 감지 및 크롭 (90% 꽉 차게)
        result_img = crop_to_ring_area(result_img)
        
        # 1000x1300으로 리사이즈
        result_img = resize_to_thumbnail(result_img)
        
        # 결과 인코딩 (padding 제거)
        buffered = BytesIO()
        result_img.save(buffered, format="PNG", quality=95)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Padding 제거
        img_base64 = img_base64.rstrip('=')
        
        return {
            "output": {
                "thumbnail": img_base64,
                "original_size": original_size,
                "thumbnail_size": (1000, 1300),
                "frame_removed": frame_mask is not None and frame_mask.any(),
                "status": "success"
            }
        }
        
    except Exception as e:
        logger.error(f"Handler error: {str(e)}")
        return {
            "output": {
                "error": str(e),
                "status": "failed"
            }
        }

runpod.serverless.start({"handler": handler}))
    return bool(base64_pattern.match(s[:100]))  # 처음 100자만 체크

def decode_image_data(image_data):
    """
    이미지 데이터 디코딩 (URL 또는 base64)
    """
    if image_data.startswith('http'):
        # URL인 경우
        logger.info(f"Fetching image from URL: {image_data}")
        response = requests.get(image_data)
        return response.content
    else:
        # Base64인 경우
        # data:image/png;base64, 접두사 제거
        if image_data.startswith('data:'):
            image_data = image_data.split(',', 1)[1]
        
        # 4가지 디코딩 시도
        for method in range(4):
            try:
                if method == 0:
                    # Direct decode
                    return base64.b64decode(image_data)
                elif method == 1:
                    # Add padding
                    padded = image_data + '=' * (4 - len(image_data) % 4)
                    return base64.b64decode(padded)
                elif method == 2:
                    # URL-safe decode
                    return base64.urlsafe_b64decode(image_data)
                elif method == 3:
                    # Force padding
                    padded = image_data + '==='
                    return base64.b64decode(padded)
            except Exception as e:
                continue
        
        raise ValueError("Failed to decode base64 image")

def handler(job):
    """
    RunPod handler for thumbnail generation with black frame removal
    """
    try:
        job_input = job['input']
        
        # 이미지 데이터 찾기
        image_data_str = find_image_data(job_input)
        if not image_data_str:
            # 디버깅을 위해 사용 가능한 키들 표시
            logger.error(f"No image found. Available keys: {list(job_input.keys())}")
            if job_input:
                # 첫 번째 키의 샘플 보여주기
                first_key = list(job_input.keys())[0]
                sample = str(job_input[first_key])[:200]
                logger.error(f"Sample data from '{first_key}': {sample}")
            raise ValueError("No image provided")
        
        # 이미지 디코딩
        image_data = decode_image_data(image_data_str)
        
        # 이미지 열기
        img = Image.open(BytesIO(image_data))
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        
        # 원본 크기 저장
        original_size = img.size
        
        # 검은색 프레임 감지
        img_array = np.array(img)
        frame_mask = detect_black_frame(img_array)
        
        if frame_mask is not None and frame_mask.any():
            logger.info("Black frame detected, removing...")
            
            # Replicate API로 인페인팅 시도
            if REPLICATE_API_TOKEN:
                result_img = remove_black_frame_with_inpainting(img, frame_mask)
            else:
                logger.warning("Replicate API token not found, using simple removal")
                result_img = simple_frame_removal(img, frame_mask)
        else:
            logger.info("No black frame detected")
            result_img = img
        
        # enhance와 동일한 색감 보정 적용
        result_img = apply_enhancement(result_img)
        
        # 웨딩링 영역 감지 및 크롭 (90% 꽉 차게)
        result_img = crop_to_ring_area(result_img)
        
        # 1000x1300으로 리사이즈
        result_img = resize_to_thumbnail(result_img)
        
        # 결과 인코딩 (padding 제거)
        buffered = BytesIO()
        result_img.save(buffered, format="PNG", quality=95)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Padding 제거
        img_base64 = img_base64.rstrip('=')
        
        return {
            "output": {
                "thumbnail": img_base64,
                "original_size": original_size,
                "thumbnail_size": (1000, 1300),
                "frame_removed": frame_mask is not None and frame_mask.any(),
                "status": "success"
            }
        }
        
    except Exception as e:
        logger.error(f"Handler error: {str(e)}")
        return {
            "output": {
                "error": str(e),
                "status": "failed"
            }
        }

runpod.serverless.start({"handler": handler})
