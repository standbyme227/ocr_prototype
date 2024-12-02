# ml_backend/app.py 파일에 아래 코드를 추가합니다.

import os
import json
import logging
import threading
import requests
from flask import Flask, jsonify, request
from ultralytics import YOLO
from label_studio_ml.model import LabelStudioMLBase

# 설정 클래스
class Config:
    API_TOKEN = os.getenv("LABEL_STUDIO_API_TOKEN")
    LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
    MODEL_DIR = os.path.join(PROJECT_DIR, "volumes", "models")
    TRAIN_DATA_DIR = os.path.join(PROJECT_DIR, "volumes", "train_data")
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.environ["ULTRALYTICS_CACHE"] = MODEL_DIR

# 파일 잠금 객체 생성 (동시성 문제 해결을 위해)
file_lock = threading.Lock()

# Flask 앱 초기화
app = Flask(__name__)

class YOLOBackend(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 모델 캐시 딕셔너리 초기화
        self.models = {}
        
        # 모델 이름 매핑 딕셔너리
        self.model_name_dict = {
            'lidl': 'lidl_model.pt',
            'tesco': 'tesco_model.pt',
            'kaufland': 'kaufland_model.pt',
            'billa': 'billa_model.pt'
        }
    
    def predict(self, tasks, **kwargs):
        """
        Label Studio가 태스크를 전달하면, YOLO로 예측을 수행하고 결과를 반환합니다.
        """
        predictions = []
        tasks_list = tasks.get("tasks", [])
        
        # 입력 값 검증
        if not tasks_list:
            logging.error("No tasks provided for prediction.")
            return predictions

        # 프로젝트 이름 가져오기
        project_name = kwargs.get('project')
        selected_model_name = self.get_model_name(project_name)

        # 모델 로드 또는 캐시에서 가져오기
        model = self.get_or_load_model(selected_model_name)
        if model is None:
            # 모델 로드 실패 시 빈 결과 반환
            return predictions
        
        # 각 태스크 처리
        for task in tasks_list:
            task_prediction = self.process_task(task, model)
            if task_prediction:
                predictions.append(task_prediction)
        
        return predictions

    def get_model_name(self, project_name):
        """
        프로젝트 이름에 따라 모델 이름을 결정합니다.
        """
        if not project_name:
            logging.warning("Project name not found. Using default model.")
            return 'best.pt'  # 기본 모델 이름
        else:
            project_name = project_name.lower()
            for key, model_name in self.model_name_dict.items():
                if key in project_name:
                    return model_name
            logging.info("No specific model found for project. Using default model.")
            return 'best.pt'  # 기본 모델 이름

    def get_or_load_model(self, model_name):
        """
        모델을 캐시에서 가져오거나 로드합니다.
        """
        if model_name not in self.models:
            model_path = os.path.join(Config.MODEL_DIR, model_name)
            if not os.path.exists(model_path):
                logging.error(f"Model not found at {model_path}. Please download it.")
                return None
            self.models[model_name] = YOLO(model_path)
            logging.info(f"Loaded model: {model_name}")
        else:
            logging.info(f"Using cached model: {model_name}")
        return self.models[model_name]

    def process_task(self, task, model):
        """
        단일 태스크를 처리하여 예측 결과를 생성합니다.
        """
        task_id = task.get("id")  # 태스크 ID
        image_path = task.get("data", {}).get("image")  # 이미지 경로
        if not image_path:
            logging.warning(f"No image path provided for task ID {task_id}.")
            return None
        
        # 이미지 경로 수정 및 확인
        corrected_path = self.get_corrected_image_path(image_path)
        if not corrected_path:
            logging.warning(f"Image path not found or invalid: {image_path}")
            return None
        
        # YOLO 예측 수행
        results = self.perform_prediction(model, corrected_path)
        if results is None:
            return None

        # 예측 결과 변환
        task_predictions = self.convert_predictions(results, corrected_path)
        if task_predictions is None:
            return None

        # 평균 신뢰도 계산
        avg_score = self.calculate_average_score(results)
        
        # 태스크별 결과 반환
        return {
            "task": task_id,
            "result": task_predictions,
            "score": avg_score,
            "model_version": model.names[0],  # 모델 버전 (필요에 따라 수정)
        }

    def get_corrected_image_path(self, image_path):
        """
        이미지 경로를 수정하고 존재 여부를 확인합니다.
        """
        corrected_path = image_path.replace('/data/', '/app/data/media/', 1)  # 경로 수정
        if not os.path.exists(corrected_path):
            return None
        return corrected_path

    def perform_prediction(self, model, image_path):
        """
        YOLO 모델을 사용하여 예측을 수행합니다.
        """
        try:
            results = model(image_path)
            return results
        except Exception as e:
            logging.error(f"Error during model prediction: {e}")
            return None

    def convert_predictions(self, results, image_path):
        """
        YOLO 예측 결과를 Label Studio 형식으로 변환합니다.
        """
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                image_width, image_height = img.size
        except Exception as e:
            logging.error(f"Error reading image dimensions: {e}")
            return None

        task_predictions = []
        for result in results:
            for i, box in enumerate(result.boxes.xyxy):
                x_min, y_min, x_max, y_max = map(float, box)  # Bounding box 좌표
                width = (x_max - x_min) / image_width * 100  # YOLO 좌표 -> % 변환
                height = (y_max - y_min) / image_height * 100
                x = x_min / image_width * 100
                y = y_min / image_height * 100
                
                score = float(result.boxes.conf[i])  # 신뢰도 점수
                
                if score < 0.985:
                # if score < 0.8:
                    continue

                task_predictions.append({
                    "from_name": "label",  # 라벨 구성 이름
                    "to_name": "image",   # 이미지 구성 이름
                    "type": "rectanglelabels",  # 박스 타입
                    "value": {
                        "x": x,  # x 좌표
                        "y": y,  # y 좌표
                        "width": width,  # 너비
                        "height": height,  # 높이
                        "rotation": 0,  # 기본 회전값
                        "rectanglelabels": ["item_information"],  # 클래스 이름
                    },
                    "score": score,  # 신뢰도
                })
        return task_predictions

    def calculate_average_score(self, results):
        """
        예측 결과의 평균 신뢰도를 계산합니다.
        """
        all_scores = []
        for result in results:
            if result.boxes.conf.numel() > 0:
                all_scores.extend(result.boxes.conf.tolist())
        if all_scores:
            return sum(all_scores) / len(all_scores)
        else:
            return 0.0
        
    def update_with_latest_data(self, existing_data, new_data):
        """
        기존 데이터에 새로운 데이터를 병합하면서 중복된 키는 최신 값으로 덮어씌운다.
        Args:
            existing_data (dict): 기존 데이터
            new_data (dict): 새롭게 추가할 데이터
        Returns:
            dict: 최신 데이터만 남은 병합된 데이터
        """
        for key, value in new_data.items():
            existing_data[key] = value  # 동일한 키가 있으면 최신 데이터로 덮어씌움
        return existing_data


    def fit(self, tasks, workdir=None, **kwargs):
        """
        Label Studio가 학습 요청을 보내면 데이터를 저장
        """
        try:
            # 프로젝트 이름 가져오기
            project_name = kwargs.get('project_name')
            if not project_name:
                logging.error("Project name not provided.")
                return {"error": "Project name is required."}

            project_name = project_name.lower().replace(" ", "_")
            project_id = kwargs.get('project_id')

            annotations = tasks.get("annotation", {})
            annotation_id = str(annotations.get("id"))

            if not annotation_id:
                logging.error("Annotation ID is missing in the data.")
                return {"error": "Annotation ID is missing in the data."}

            # 파일 경로 설정
            train_data_path = os.path.join(Config.TRAIN_DATA_DIR, f"{project_name}_{project_id}_train_data.json")

            with file_lock:
                # 기존 데이터 로드
                if os.path.exists(train_data_path):
                    with open(train_data_path, "r") as f:
                        existing_data = json.load(f)
                else:
                    existing_data = {}

                # 새로운 데이터 생성
                new_data = {
                    annotation_id: {
                        "task_id": tasks.get("task", {}).get("id"),
                        "image_path": tasks.get("task", {}).get("data", {}).get("image"),
                        "annotations": annotations.get("result", [])
                    }
                }

                # 최신 데이터로 병합
                updated_data = self.update_with_latest_data(existing_data, new_data)

                # 데이터 저장
                with open(train_data_path, "w") as f:
                    json.dump(updated_data, f, indent=4)

            return {"status": "Training data saved successfully!", "file_path": train_data_path}

        except Exception as e:
            logging.error(f"Error saving training data: {str(e)}")
            return {"error": str(e)}

# 공통 함수: 프로젝트 정보 가져오기 (중복 코드 제거)
def get_project_info(tasks=None, train_tasks=None):
    if not tasks and not train_tasks:
        logging.error("No tasks provided.")
        raise ValueError("No tasks provided.")
    
    if tasks:
        task = tasks.get("tasks", [])[0]
        project_id = task.get("project")
        if not project_id:
            logging.error("Project ID not found in task.")
            raise ValueError("Project ID is required.")
        project_name = get_project_name(project_id, Config.LABEL_STUDIO_URL, Config.API_TOKEN)
    else:
        project = train_tasks.get("project")
        project_name = project.get("title")
        project_id = project.get("id")
        
    return project_id, project_name 

def get_project_name(project_id, label_studio_url, api_key):
    """
    Label Studio API에서 프로젝트 이름을 조회합니다.

    Args:
        project_id (int): 프로젝트 ID
        label_studio_url (str): Label Studio URL
        api_key (str): Label Studio API 토큰

    Returns:
        str: 프로젝트 이름

    Raises:
        ValueError: 프로젝트 이름 조회 실패 시
    """
    url = f"{label_studio_url}/api/projects/{project_id}"
    headers = {"Authorization": f"Token {api_key}"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json().get("title")
    else:
        logging.error(f"Failed to fetch project name: {response.text}")
        raise ValueError(f"Failed to fetch project name: {response.text}")

# Flask 엔드포인트 설정
backend = YOLOBackend()

@app.route("/predict", methods=["POST"])
def predict():
    """
    Label Studio에서 예측 요청을 처리
    """
    try:
        tasks = request.json

        # 입력 값 검증
        if not tasks or "tasks" not in tasks:
            logging.error("Invalid request format: 'tasks' key is missing.")
            return jsonify({"error": "Invalid request format: 'tasks' key is missing."}), 400

        # 프로젝트 정보 가져오기
        project_id, project_name = get_project_info(tasks=tasks)
        predictions = backend.predict(tasks, project=project_name)
        return jsonify({"results": predictions}), 200
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/train", methods=["POST"])
def train():
    """
    Label Studio에서 학습 요청을 처리
    """
    try:
        tasks = request.json

        # 프로젝트 정보 가져오기
        project_id, project_name = get_project_info(train_tasks=tasks)
        result = backend.fit(tasks, project_name=project_name, project_id=project_id)
        if "error" in result:
            return jsonify(result), 400
        return jsonify(result), 200
    except Exception as e:
        logging.error(f"Training error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/webhook", methods=["POST"])
def webhook():
    """
    Label Studio 웹훅 엔드포인트.
    Label Studio에서 전달받은 이벤트 데이터를 처리.
    """
    try:
        data = request.json  # 웹훅으로 전송된 JSON 데이터를 가져옴

        # 이벤트 액션 확인
        action = data.get("action")
        project = data.get("project", {})
        project_id = project.get("id")
        project_name = project.get("title", "default_project")

        if not action:
            logging.warning("No action specified in webhook data.")
            return jsonify({"error": "No action specified"}), 400

        # 로그로 받은 데이터를 확인
        logging.info(f"Webhook triggered: {action} for project {project_name} (ID: {project_id})")

        if action == "START_TRAINING":
            # 학습 시작 신호일 경우, 필요한 데이터 저장/처리
            logging.info(f"Training started for project {project_name} (ID: {project_id})")

            # 추가 작업이 필요하다면 여기에 구현
            return jsonify({"status": "Training started", "project_id": project_id}), 200

        elif action == "TASK_SUBMITTED":
            # 태스크 제출 이벤트 처리
            logging.info(f"Task submitted for project {project_name} (ID: {project_id})")

            # 추가 작업이 필요하다면 여기에 구현
            return jsonify({"status": "Task submitted processed", "project_id": project_id}), 200

        else:
            # 알 수 없는 액션에 대한 로그 기록
            logging.warning(f"Unhandled action: {action}")
            return jsonify({"status": "Unhandled action", "action": action}), 400

    except Exception as e:
        logging.error(f"Error in webhook: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Health Check 엔드포인트
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "UP"}), 200

# Label Studio 초기화 엔드포인트
@app.route("/setup", methods=["POST"])
def setup():
    """
    Label Studio에서 설정 데이터를 받아 초기화합니다.
    """
    try:
        # Label Studio가 전달한 요청 데이터에서 `schema` 추출
        config = request.json.get("schema")
        if config:
            logging.info("Label schema received and initialized.")
            # schema를 이용한 추가 작업 (필요 시)
        else:
            logging.warning("No schema provided.")
            return jsonify({"error": "No schema provided"}), 400

        # 성공적으로 처리되었음을 응답
        return jsonify({"message": "Setup successful"}), 200
    except Exception as e:
        logging.error(f"Error during setup: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Flask 앱 실행
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9090, debug=True)