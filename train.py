import os
import json
import yaml
from ultralytics import YOLO
import shutil

# 학습 데이터 이름
TRAIN_DATA_NAME = "project_3_train_data"
file_extension = ".json"

# 경로 설정
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 디렉토리
TRAIN_DATA_PATH = os.path.join(PROJECT_DIR, 'volumes', 'train_data', TRAIN_DATA_NAME + file_extension)  # 학습 데이터 경로
COMPLETED_DATA_DIR = os.path.join(PROJECT_DIR, 'volumes', 'completed_data')  # 학습 완료된 데이터 저장 폴더
MODEL_PATH = os.path.join(PROJECT_DIR, 'volumes', 'models', 'best.pt')
# MODEL_PATH = os.path.join(PROJECT_DIR, 'volumes', 'models', 'yolo11m.pt')

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

    # YOLO 형식 데이터 준비
    image_dir = os.path.join(PROJECT_DIR, 'volumes', 'train_data', TRAIN_DATA_NAME, 'images')
    label_dir = os.path.join(PROJECT_DIR, 'volumes', 'train_data', TRAIN_DATA_NAME, 'labels')
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    task_id_list = labeled_data.keys()
    for task_id in task_id_list:
        item = labeled_data[task_id]
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

                # YOLO 형식으로 좌표 변환
                x_center = (value['x'] + value['width'] / 2) / 100.0  # %를 비율로 변환
                y_center = (value['y'] + value['height'] / 2) / 100.0
                width = value['width'] / 100.0
                height = value['height'] / 100.0

                # 클래스 ID 설정 (여기선 기본값 가정)
                class_id = 0  # "item_information"이 단일 클래스인 경우

                # YOLO 형식으로 저장
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

    # YAML 파일 생성
    dataset_yaml_path = os.path.join(PROJECT_DIR, 'volumes', 'train_data', TRAIN_DATA_NAME, 'dataset.yaml')
    dataset = {
        "path": os.path.join(PROJECT_DIR, 'volumes', 'train_data', TRAIN_DATA_NAME),
        "train": "images",
        "val": "images",  # 검증 데이터가 같은 폴더에 있다고 가정
        "names": ["item_information"]  # 클래스 이름
    }
    with open(dataset_yaml_path, 'w') as yaml_file:
        yaml.dump(dataset, yaml_file)

    return dataset_yaml_path

# def train_model(train_data_path):
#     """
#     yolo11m 모델 학습 수행 및 학습 완료 처리
#     """
#     # 데이터셋 준비
#     dataset = prepare_yolo_dataset(train_data_path)
#     print("Dataset prepared.")
    
#     # 학습 수행
#     # model.train(data=dataset, epochs=10, imgsz=640)
    
#     # 유사도가 높은 이미지를 학습한다면 freeze를 늘린다.
#     model.train(data=dataset, epochs=10, imgsz=640, freeze=10)
#     print("Model training completed!")
    
#     # 학습 완료 후 데이터 처리
#     mark_data_as_completed(train_data_path)

def train_model(train_data_path):
    """
    YOLO 모델 학습 수행 및 학습 완료 처리
    """
    # 데이터셋 준비
    dataset = prepare_yolo_dataset(train_data_path)
    print("Dataset prepared.")
    
    # MPS 장치 확인
    import torch
    if torch.backends.mps.is_available():
        device = 'mps'
        print("Using MPS device for training.")
    else:
        device = 'cpu'
        print("MPS not available, using CPU.")
    
    # 학습 수행
    model.train(
        data=dataset,
        epochs=50,
        imgsz=640,
        batch=16,
        workers=4,
        device=device,
        freeze=0,
        lr0=0.001,
        optimizer='Adam',
        cos_lr=True,
        patience=5
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
    # 학습 완료된 데이터 디렉토리 생성
    os.makedirs(COMPLETED_DATA_DIR, exist_ok=True)

    # 파일 이름에 "_completed" 추가
    base_name = os.path.basename(data_path)
    completed_name = base_name.replace('.json', '_completed.json')
    completed_dir_path = os.path.join(COMPLETED_DATA_DIR, TRAIN_DATA_NAME)
    completed_path = os.path.join(COMPLETED_DATA_DIR, TRAIN_DATA_NAME, completed_name)
    os.makedirs(completed_dir_path, exist_ok=True)

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