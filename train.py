import os
import json
import yaml
from ultralytics import YOLO
import shutil

# env 파일 로드
from dotenv import load_dotenv
load_dotenv()

# 환경 변수 설정
TRAIN_DATA_NAME = os.getenv("TRAIN_DATA_NAME")
file_extension = ".json"

# 경로 설정
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 디렉토리
RESULT_PROJECT_DIR = os.path.join(PROJECT_DIR, 'results')  # 결과 저장 디렉토리
TRAIN_DATA_PATH = os.path.join(PROJECT_DIR, 'volumes', 'train_data', TRAIN_DATA_NAME + file_extension)  # 학습 데이터 경로
COMPLETED_DATA_DIR = os.path.join(PROJECT_DIR, 'volumes', 'completed_data')  # 학습 완료된 데이터 저장 폴더
# MODEL_PATH = os.path.join(PROJECT_DIR, 'volumes', 'models', 'yolo11m.pt')

model_dict = {
    "kaufland" : "kaufland_model.pt",
    "tesco" : "tesco_model.pt",
    "lidl" : "lidl_model.pt",
    "billa" : "billa_model.pt",
}

# 학습데이터 이름을 확인해서 모델을 선택한다.
def select_model(train_data_name):
    if "kaufland" in train_data_name:
        return model_dict["kaufland"]
    elif "tesco" in train_data_name:
        return model_dict["tesco"]
    elif "lidl" in train_data_name:
        return model_dict["lidl"]
    elif "billa" in train_data_name:
        return model_dict["billa"]
    else:
        return "yolo11m.pt"

model_name = select_model(TRAIN_DATA_NAME)
MODEL_PATH = os.path.join(PROJECT_DIR, 'volumes', 'models', model_name)

# YOLO 모델 초기화
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Ensure it is downloaded.")
model = YOLO(MODEL_PATH)

def prepare_yolo_dataset(train_data_path):
    """
    YOLO 학습을 위해 데이터셋 준비
    Args:
        train_data_path (str): labeled_data.json 경로
    Returns:
        str: YOLO 학습 데이터 YAML 파일 경로
    """
    with open(train_data_path, 'r') as f:
        labeled_data = json.load(f)

    # 클래스 이름 추출
    class_names_set = set()
    for task_id, item in labeled_data.items():
        annotations = item['annotations']
        for annotation in annotations:
            label = annotation['value'].get('rectanglelabels', [])[0]
            class_names_set.add(label)

    # 클래스 이름을 리스트로 변환하고 정렬 (일관된 클래스 ID를 위해)
    class_names = sorted(list(class_names_set))
    class_name_to_id = {name: idx for idx, name in enumerate(class_names)}

    # YOLO 형식 데이터 준비
    image_dir = os.path.join(PROJECT_DIR, 'volumes', 'train_data', TRAIN_DATA_NAME, 'images')
    label_dir = os.path.join(PROJECT_DIR, 'volumes', 'train_data', TRAIN_DATA_NAME, 'labels')
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    task_id_list = labeled_data.keys()
    for task_id in task_id_list:
        item = labeled_data[task_id]
        annotations = item['annotations']
        if not annotations:
            print(f"Skipping image with no annotations: {item['image_path']}")
            continue  # 어노테이션이 없는 이미지는 데이터셋에서 제외
        
        image_path = item['image_path']
        
        # 경로 수정 로직: '/data/' -> '/data/media/'
        if image_path.startswith('/data/'):
            corrected_path = image_path.replace('/data/', '/data/media/', 1)
        else:
            corrected_path = image_path  # 수정이 필요 없는 경우 그대로 사용
        
        img_ab_path = os.path.join(PROJECT_DIR, corrected_path.lstrip('/'))
        annotations = item['annotations']
                
        destination_path = os.path.join(image_dir, os.path.basename(image_path))
        
        if os.path.exists(destination_path):
            continue

        # 이미지 경로 처리 (YOLO에 맞게 이미지 이동)
        os.link(img_ab_path, destination_path)
        image_extension = os.path.splitext(image_path)[1]

        # 라벨 파일 생성
        label_file = os.path.join(label_dir, os.path.basename(image_path).replace(image_extension, '.txt'))
        with open(label_file, 'w') as f:
            for annotation in annotations:
                value = annotation['value']

                # 클래스 이름 및 ID 가져오기
                label = value.get('rectanglelabels', [])[0]
                class_id = class_name_to_id[label]

                # YOLO 형식으로 좌표 변환
                x_center = (value['x'] + value['width'] / 2) / 100.0  # %를 비율로 변환
                y_center = (value['y'] + value['height'] / 2) / 100.0
                width = value['width'] / 100.0
                height = value['height'] / 100.0

                # YOLO 형식으로 저장
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

    # YAML 파일 생성
    dataset_yaml_path = os.path.join(PROJECT_DIR, 'volumes', 'train_data', TRAIN_DATA_NAME, 'dataset.yaml')
    dataset = {
        "path": os.path.join(PROJECT_DIR, 'volumes', 'train_data', TRAIN_DATA_NAME),
        "train": "images",
        "val": "images",  # 검증 데이터가 같은 폴더에 있다고 가정
        "nc": len(class_names),  # 클래스 수
        "names": class_names  # 클래스 이름 리스트
    }
    with open(dataset_yaml_path, 'w') as yaml_file:
        yaml.dump(dataset, yaml_file)

    return dataset_yaml_path

def train_model(train_data_path):
    """
    YOLO 모델 학습 수행 및 학습 완료 처리
    """
    from ultralytics import YOLO
    import yaml

    # 데이터셋 준비
    dataset = prepare_yolo_dataset(train_data_path)
    print("Dataset prepared.")

    # 데이터셋 YAML 파일에서 클래스 정보 읽기
    with open(dataset, 'r') as f:
        data_yaml = yaml.safe_load(f)
        
    class_names = data_yaml['names']
    num_classes = data_yaml['nc']

    # MPS 장치 확인
    import torch
    if torch.backends.mps.is_available():
        device = 'mps'
        print("Using MPS device for training.")
    else:
        device = 'cpu'
        print("MPS not available, using CPU.")

    # YOLO 모델 초기화 (클래스 수 설정)
    model.model.nc = num_classes  # 클래스 수 설정
    model.model.names = class_names  # 클래스 이름 설정

    # 학습 수행
    model.train(
        data=dataset,
        device=device,  # 장치 설정 (필요 시 'cpu'로 변경)
        
        freeze=7,
        
        epochs=10,
        # patience=3,
        imgsz=640,      # 이미지 크기 감소
        
        batch=4,        # 배치 크기 감소
        workers=4,      # 워커 수 조정

        # optimizer='Adam',
        cos_lr=True,
        
        project=RESULT_PROJECT_DIR,
        name=TRAIN_DATA_NAME,
    )
    print("Model training completed!")

    # 학습 완료 후 데이터 처리
    mark_data_as_completed(train_data_path)

def mark_data_as_completed(data_path):
    """
    학습 완료된 데이터를 파일 이동 또는 이름 변경
    Args:
        data_path (str): 학습 데이터 파일 경로
    """
    # 학습 완료 데이터 디렉토리 확인 및 생성
    if not os.path.exists(COMPLETED_DATA_DIR):
        os.makedirs(COMPLETED_DATA_DIR, exist_ok=True)

    # 학습 완료 데이터 처리
    base_name = os.path.basename(data_path)
    completed_dir_path = os.path.join(COMPLETED_DATA_DIR, TRAIN_DATA_NAME)
    os.makedirs(completed_dir_path, exist_ok=True)
    
    completed_path = os.path.join(COMPLETED_DATA_DIR, TRAIN_DATA_NAME, base_name)

    # 데이터 파일 이동
    shutil.move(data_path, completed_path)
    print(f"Training data moved to: {completed_path}")

def check_and_confirm():
    """
    학습 시작 전에 데이터 및 설정을 확인하고 사용자로부터 확인을 받는 함수
    Returns:
        bool: 사용자가 'confirm'을 입력하면 True, 그렇지 않으면 False
    """
    print("===== Training Configuration =====")
    print(f"Training Data Name: {TRAIN_DATA_NAME}")
    print(f"Training Data Path: {TRAIN_DATA_PATH}")
    print(f"Model Path: {MODEL_PATH}")
    print(f"Completed Data Directory: {COMPLETED_DATA_DIR}")
    print("\nChecking files and directories...")

    # 파일 및 디렉토리 존재 여부 확인
    errors = False
    if not os.path.exists(TRAIN_DATA_PATH):
        print(f"[Error] Training data file not found at {TRAIN_DATA_PATH}")
        errors = True
    else:
        print(f"[OK] Training data file found at {TRAIN_DATA_PATH}")

    if not os.path.exists(MODEL_PATH):
        print(f"[Error] Model file not found at {MODEL_PATH}")
        errors = True
    else:
        print(f"[OK] Model file found at {MODEL_PATH}")

    # 최종 저장 위치 확인
    if not os.path.exists(COMPLETED_DATA_DIR):
        print(f"[Info] Completed data directory will be created at {COMPLETED_DATA_DIR}")
    else:
        print(f"[OK] Completed data directory exists at {COMPLETED_DATA_DIR}")

    # 오류가 있는 경우 스크립트 종료
    if errors:
        print("\nOne or more errors detected. Please fix them before proceeding.")
        return False

    print("\nAll checks passed.")
    print("Please review the above information carefully.")
    confirm = input("If everything is correct, type 'confirm' to proceed with training: ")

    if confirm.lower() == 'confirm':
        print("Confirmation received. Starting training...")
        return True
    else:
        print("Training cancelled by user.")
        return False

# 학습 실행 (로컬에서 실행)
if __name__ == "__main__":
    # 사용자 확인 절차 실행
    if check_and_confirm():
        train_model(TRAIN_DATA_PATH)
    else:
        print("Exiting script.")