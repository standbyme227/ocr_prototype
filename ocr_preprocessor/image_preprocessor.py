import cv2
import numpy as np
import os

# 전처리 함수
def preprocess_image(image_path, output_dir):
    try:
        # 이미지 읽기 (흑백으로 로드)
        # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # 이미지 읽기
        image = cv2.imread(image_path)
        image_name = image_path.split("/")[-1]
        
        if image is None:
            raise ValueError(f"Failed to load image from {image_path}")

        # # Step 1: 해상도 업스케일링
        result = cv2.resize(image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # 그레이스케일 변환
        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        # CLAHE 적용
        clahe = cv2.createCLAHE(
                clipLimit=2.0, 
                tileGridSize=(5,5)
            )
        result = clahe.apply(result)

        # 언샤프 마스크 적용
        gaussian = cv2.GaussianBlur(result, (9, 9), 10.0)
        result = cv2.addWeighted(result, 1.5, gaussian, -0.5, 0, result)

        # 출력 경로 설정
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        image_name = image_name.split(".")[0]
        new_image_name = f"{image_name}(processed).png"
        
        output_path = os.path.join(output_dir, new_image_name)
        cv2.imwrite(output_path, result)
        print(f"Processed image saved at {output_path}")
        
        return new_image_name
    
    except Exception as e:
        raise e