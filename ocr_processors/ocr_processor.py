from .ocr_language_manager import OCRLanguageManager
import cv2
import numpy as np

# PaddleOCR 전역 설정
ocr_manager = OCRLanguageManager()

def group_text_by_distance(ocr_results, distance_threshold=50):
    """
    OCR 결과를 박스 간 거리 기반으로 그룹화하여 텍스트를 병합.
    """
    grouped_texts = []  # 그룹화된 텍스트 리스트
    current_group = {"text": "", "center": None}  # 현재 그룹 초기화

    for result in ocr_results:
        text = result["text"]
        box = result["box"]
        # 박스 중심 계산
        x_min, y_min, x_max, y_max = box[0][0], box[0][1], box[2][0], box[2][1]
        box_center = ((x_min + x_max) / 2, (y_min + y_max) / 2)

        # 첫 그룹 초기화
        if current_group["center"] is None:
            current_group["text"] = text
            current_group["center"] = box_center
        else:
            # 현재 그룹과 새 박스 중심 간 거리 계산
            cx, cy = current_group["center"]
            bx, by = box_center
            distance = np.sqrt((cx - bx) ** 2 + (cy - by) ** 2)

            if distance < distance_threshold:
                # 그룹 내 텍스트 병합
                current_group["text"] += f" {text}."
                current_group["center"] = (
                    (cx + bx) / 2,
                    (cy + by) / 2
                )
            else:
                # 그룹 완료, 새 그룹 시작
                grouped_texts.append(current_group["text"].strip())
                current_group = {"text": f"{text}.", "center": box_center}

    # 마지막 그룹 추가
    if current_group["text"]:
        grouped_texts.append(current_group["text"].strip())

    return grouped_texts


def process_ocr(image_path=None, image=None, language="sk", group=False):
    """
    이미지 경로 또는 이미지 데이터를 입력받아 OCR 처리 결과를 반환하는 함수.
    """

    if not image_path and image is None:
        raise ValueError("No image provided")

    # 언어별 OCR 모델 가져오기
    ocr_model = ocr_manager.get_ocr_model(language)

    # 이미지 로드 (image_path 우선)
    if image is None:
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image from path: {image_path}")
        except Exception as e:
            raise ValueError(f"Error loading image from path: {e}")
        
    # OCR 처리
    try:
        results = ocr_model.ocr(img=image, cls=True)
    except Exception as e:
        raise RuntimeError(f"OCR processing failed: {e}")

    # OCR 결과 변환
    response = []
    for line in results[0]:
        text, confidence = line[1]
        box = line[0]
        response.append({
            "text": text,
            "confidence": confidence,
            "box": box
        })

    # 그룹화 로직 추가
    if group:
        grouped_texts = group_text_by_distance(response)
        return grouped_texts
    else:
        return response