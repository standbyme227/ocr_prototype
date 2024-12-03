from flask import Flask, request, jsonify
from batch_image_preprocessor import batch_processor

app = Flask(__name__)

@app.route("/preprocess", methods=["POST"])
def preprocess_images():
    data = request.get_json()
    input_folder = data.get("input_folder")
    if not input_folder:
        return jsonify({"error": "No input folder provided"}), 400

    try:
        output_folder = batch_processor(input_folder)
        return jsonify({"message": "Preprocessing complete", "output_folder": output_folder}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)