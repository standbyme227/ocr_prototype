import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify
from ocr_processors.ocr_processor import process_ocr

app = Flask(__name__)

@app.route("/ocr", methods=["POST"])
def process_image_request():
    """
    API 요청에서 이미지 경로(단일 또는 폴더)와 언어 정보를 받아 OCR 처리 결과를 반환.
    """
    try:
        data = request.get_json()
        language = data.get("language", "sk")
        image_path = data.get("image_path")
        folder_path = data.get("folder_path")
        group = data.get("group", False)

        if not image_path and not folder_path:
            return jsonify({"error": "No image path or folder path provided"}), 400

        results = []

        if folder_path:
            # 폴더 내 모든 이미지 처리
            if not os.path.isdir(folder_path):
                return jsonify({"error": "Provided folder path is invalid"}), 400

            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if os.path.isfile(file_path):
                    try:
                        ocr_result = process_ocr(image_path=file_path, language=language, group=group)
                        results.append({"image": file_name, "ocr_result": ocr_result})
                    except Exception as e:
                        results.append({"image": file_name, "error": str(e)})

            # 결과를 JSON 파일로 저장
            output_path = os.path.join(folder_path, "ocr_results.json")
            with open(output_path, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

            return jsonify({"message": "OCR processing completed for folder", "output_path": output_path}), 200

        elif image_path:
            # 단일 이미지 처리
            if not os.path.isfile(image_path):
                return jsonify({"error": "Provided image path is invalid"}), 400

            ocr_result = process_ocr(image_path=image_path, language=language, group=group)
            return jsonify(
                    {
                        "data": ocr_result,
                        "output_folder": os.path.dirname(image_path)
                    },
                ), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except RuntimeError as re:
        return jsonify({"error": str(re)}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8082)  # PaddleOCR API 실행