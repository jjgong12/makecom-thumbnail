import runpod
import base64
from io import BytesIO
from PIL import Image, ImageOps, ImageDraw, ImageFont
import requests

# 색상 매핑
COLOR_MAPPING = {
    'yellow_gold': {'bg': (255, 248, 220), 'text': (218, 165, 32)},
    'rose_gold': {'bg': (255, 228, 225), 'text': (183, 110, 121)}, 
    'white_gold': {'bg': (245, 245, 245), 'text': (192, 192, 192)},
    'white': {'bg': (255, 255, 255), 'text': (200, 200, 200)}
}

def create_thumbnail(base64_image, metal_color='yellow_gold', size=(800, 800)):
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
        metal_color = job_input.get('metal_color', 'yellow_gold')
        
        if not base64_image:
            return {"output": {"error": "No image provided", "success": False}}
        
        # 색상 검증
        valid_colors = ['yellow_gold', 'rose_gold', 'white_gold', 'white']
        if metal_color not in valid_colors:
            metal_color = 'yellow_gold'
        
        # 썸네일 생성
        thumbnail_base64 = create_thumbnail(base64_image, metal_color)
        
        if thumbnail_base64:
            return {
                "output": {
                    "success": True,
                    "thumbnail": thumbnail_base64,
                    "metal_color": metal_color,
                    "message": "Thumbnail created successfully"
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
