import runpod
import base64
from io import BytesIO
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import numpy as np
import cv2

# 목표 썸네일 크기
THUMBNAIL_WIDTH = 1000
THUMBNAIL_HEIGHT = 1300

def detect_wedding_rings(image):
    """웨딩링 감지 및 위치 찾기"""
    # PIL to OpenCV
    img_array = np.array(image)
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # 감지를 위한 이미지 축소 (최대 1500px)
    height, width = img_array.shape[:2]
    if max(height, width) > 1500:
        scale = 1500 / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img_small = cv2.resize(img_array, (new_width, new_height))
    else:
        img_small = img_array
        scale = 1.0
    
    # 그레이스케일 변환
    gray = cv2.cvtColor(img_small, cv2.COLOR_RGB2GRAY)
    
    # 엣지 검출
    edges = cv2.Canny(gray, 50, 150)
    
    # 원형 객체 감지
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=30,
        param1=50,
        param2=30,
        minRadius=20,
        maxRadius=200
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # 원본 크기로 스케일 복원
        circles[0, :, :2] = circles[0, :, :2] / scale
        
        # 가장 큰 원들 찾기
        sorted_circles = sorted(circles[0], key=lambda x: x[2], reverse=True)
        
        # 상위 2개 원의 중심점 계산
        if len(sorted_circles) >= 2:
            center_x = (sorted_circles[0][0] + sorted_circles[1][0]) // 2
            center_y = (sorted_circles[0][1] + sorted_circles[1][1]) // 2
        else:
            center_x = sorted_circles[0][0]
            center_y = sorted_circles[0][1]
        
        return int(center_x), int(center_y), True
    
    # 원을 못 찾으면 이미지 중앙 사용
    return width // 2, height // 2, False

def detect_metal_color(image, ring_area=None):
    """웨딩링의 금속 색상 감지"""
    img_array = np.array(image)
    
    # 링 영역이 있으면 해당 부분만 분석
    if ring_area:
        x, y, w, h = ring_area
        img_array = img_array[y:y+h, x:x+w]
    
    # HSV 변환
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    # 색상별 범위 정의
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])
    
    rose_lower = np.array([0, 50, 100])
    rose_upper = np.array([10, 150, 255])
    
    white_lower = np.array([0, 0, 200])
    white_upper = np.array([180, 30, 255])
    
    # 마스크 생성
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    rose_mask = cv2.inRange(hsv, rose_lower, rose_upper)
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    
    # 픽셀 수 계산
    yellow_pixels = cv2.countNonZero(yellow_mask)
    rose_pixels = cv2.countNonZero(rose_mask)
    white_pixels = cv2.countNonZero(white_mask)
    
    # 가장 많은 색상 결정
    max_pixels = max(yellow_pixels, rose_pixels, white_pixels)
    
    if max_pixels == yellow_pixels:
        return 'yellow_gold'
    elif max_pixels == rose_pixels:
        return 'rose_gold'
    else:
        return 'white'

def create_thumbnail(base64_image, size=(THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT)):
    """썸네일 생성 함수"""
    try:
        # base64 디코드
        if ',' in base64_image:
            base64_image = base64_image.split(',')[1]
        
        image_data = base64.b64decode(base64_image)
        image = Image.open(BytesIO(image_data))
        
        # RGBA로 변환
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # 색상 설정
        colors = COLOR_MAPPING.get(metal_color, COLOR_MAPPING['yellow_gold'])
        bg_color = colors['bg']
        text_color = colors['text']
        
        # 배경 생성
        background = Image.new('RGBA', size, bg_color + (255,))
        
        # 이미지 리사이즈 (패딩 포함)
        image.thumbnail((int(size[0] * 0.8), int(size[1] * 0.8)), Image.Resampling.LANCZOS)
        
        # 중앙 배치
        x = (size[0] - image.width) // 2
        y = (size[1] - image.height) // 2
        
        # 그림자 효과
        shadow = Image.new('RGBA', size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow)
        shadow_offset = 5
        
        # 부드러운 그림자 생성
        for i in range(3):
            shadow_alpha = 30 - (i * 10)
            offset = shadow_offset + i
            shadow_draw.ellipse(
                [x - offset, y - offset, 
                 x + image.width + offset, 
                 y + image.height + offset],
                fill=(0, 0, 0, shadow_alpha)
            )
        
        # 배경에 그림자 합성
        background = Image.alpha_composite(background, shadow)
        
        # 이미지 합성
        background.paste(image, (x, y), image)
        
        # 장식 프레임
        draw = ImageDraw.Draw(background)
        
        # 얇은 테두리
        draw.rectangle(
            [10, 10, size[0]-10, size[1]-10],
            outline=text_color + (100,),
            width=1
        )
        
        # 코너 장식
        corner_size = 30
        corner_width = 2
        
        # 좌상단
        draw.line([(10, 10), (10 + corner_size, 10)], fill=text_color, width=corner_width)
        draw.line([(10, 10), (10, 10 + corner_size)], fill=text_color, width=corner_width)
        
        # 우상단
        draw.line([(size[0]-10-corner_size, 10), (size[0]-10, 10)], fill=text_color, width=corner_width)
        draw.line([(size[0]-10, 10), (size[0]-10, 10+corner_size)], fill=text_color, width=corner_width)
        
        # 좌하단
        draw.line([(10, size[1]-10-corner_size), (10, size[1]-10)], fill=text_color, width=corner_width)
        draw.line([(10, size[1]-10), (10+corner_size, size[1]-10)], fill=text_color, width=corner_width)
        
        # 우하단
        draw.line([(size[0]-10-corner_size, size[1]-10), (size[0]-10, size[1]-10)], fill=text_color, width=corner_width)
        draw.line([(size[0]-10, size[1]-10-corner_size), (size[0]-10, size[1]-10)], fill=text_color, width=corner_width)
        
        # 최종 이미지를 RGB로 변환
        final_image = Image.new('RGB', size, bg_color)
        final_image.paste(background, (0, 0), background)
        
        # base64로 인코딩 - Google Script용이므로 padding 유지
        buffer = BytesIO()
        final_image.save(buffer, format='JPEG', quality=95)
        buffer.seek(0)
        
        # padding이 있는 상태로 반환
        thumb_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return thumb_base64
        
    except Exception as e:
        print(f"썸네일 생성 오류: {str(e)}")
        return None

def handler(job):
    """RunPod handler function"""
    try:
        job_input = job['input']
        
        # 입력 받기
        base64_image = job_input.get('image')
        
        if not base64_image:
            return {"output": {"error": "No image provided", "success": False}}
        
        # 썸네일 생성 (웨딩링 감지 → 크롭 → 색상 검증 → 디테일 보정)
        thumbnail_base64, detected_color, ring_found = create_thumbnail(base64_image)
        
        if thumbnail_base64:
            return {
                "output": {
                    "success": True,
                    "thumbnail": thumbnail_base64,
                    "detected_color": detected_color,
                    "ring_found": ring_found,
                    "crop_size": f"{THUMBNAIL_WIDTH}x{THUMBNAIL_HEIGHT}",
                    "message": "Thumbnail created successfully with ring detection and detail enhancement"
                }
            }
        else:
            return {
                "output": {
                    "error": "Failed to create thumbnail",
                    "success": False
                }
            }
            
    except Exception as e:
        return {
            "output": {
                "error": str(e),
                "success": False
            }
        }

runpod.serverless.start({"handler": handler})
