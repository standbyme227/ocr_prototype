import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify
from ocr_processors.ocr_processor import process_ocr

app = Flask(__name__)

@app.route("/ocr", methods=["POST"])
def process_image_request():
    """
    API 요청에서 이미지 경로와 언어 정보를 받아 OCR 처리 결과를 반환.
    """
    try:
        data = request.get_json()
        language = data.get("language", "sk")
        image_path = data.get("image_path")
        group = data.get("group", False)

        if not image_path:
            return jsonify({"error": "No image path provided"}), 400

        # OCR 처리 호출
        ocr_results = process_ocr(image_path=image_path, language=language, group=group)
        return jsonify({"data": ocr_results}), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except RuntimeError as re:
        return jsonify({"error": str(re)}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8082)  # PaddleOCR API 실행