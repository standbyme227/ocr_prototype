from flask import Flask, request, jsonify
from paddleocr import PaddleOCR
import cv2

app = Flask(__name__)

# PaddleOCR 전역 설정
p_ocr = PaddleOCR(use_angle_cls=True, lang='sk')


def process_ocr(image_path=None, image=None, language="sk"):
    """
    이미지 경로 또는 이미지 데이터를 입력받아 OCR 처리 결과를 반환하는 함수.
    내부적으로 에러를 raise 하며, 결과를 반환.
    """
    global p_ocr

    if not image_path and image is None:
        raise ValueError("No image provided")

    # 언어 설정 변경
    if p_ocr.lang != language:
        p_ocr = PaddleOCR(use_angle_cls=True, lang=language)

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
        results = p_ocr.ocr(image, cls=True)
    except Exception as e:
        raise RuntimeError(f"OCR processing failed: {e}")

    # 결과 변환
    response = []
    for line in results[0]:
        text, confidence = line[1]
        box = line[0]
        response.append({
            "text": text,
            "confidence": confidence,
            "box": box
        })

    return response


@app.route("/ocr", methods=["POST"])
def process_image_request():
    """
    API 요청에서 이미지 경로와 언어 정보를 받아 OCR 처리 결과를 반환.
    """
    try:
        data = request.get_json()
        language = data.get("language", "sk")
        image_path = data.get("image_path")

        if not image_path:
            return jsonify({"error": "No image path provided"}), 400

        # OCR 처리 호출
        ocr_results = process_ocr(image_path=image_path, language=language)
        return jsonify({"data": ocr_results}), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except RuntimeError as re:
        return jsonify({"error": str(re)}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8082)  # PaddleOCR API 실행