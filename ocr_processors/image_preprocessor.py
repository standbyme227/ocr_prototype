import cv2
import numpy as np
import os

# 전처리 함수
def preprocess_image(image_path, output_dir):
    try:
        # 이미지 읽기 (흑백으로 로드)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image_name = image_path.split("/")[-1]

        if image is None:
            print(f"Error: Could not load image from path {image_path}")
            return

        # Step 1: 해상도 업스케일링
        resized_image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        # # Step 2-1: 잡음 제거 (선택 사항)
        # blurred_image = cv2.GaussianBlur(resized_image, (3, 3), 0)

        # Step 2-2: 이진화
        _, binary_image = cv2.threshold(resized_image, 120, 255, cv2.THRESH_BINARY)

        # # Step 3: 어댑티브 이진화 (선택 사항, 기존 이진화 대체 가능)
        # adaptive_thresh = cv2.adaptiveThreshold(
        #     binary_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 0.3
        # )

        # # Step 4: 엣지 강조 (선택 사항)
        # edges = cv2.Canny(binary_image, 100, 200)

        # # Step 5: 문자 영역 강조 (선택 사항)
        # kernel = np.ones((1, 1), np.uint8)
        # eroded_image = cv2.erode(binary_image, kernel, iterations=1)

        # # Step 6: 회색 부분을 흰색으로 변경
        # # 회색 값 범위 지정 (조정 가능)
        # lower_gray = 100  # 회색 하한값
        # upper_gray = 200  # 회색 상한값

        # # 마스크 생성 (회색 영역 찾기)
        # mask = cv2.inRange(resized_image, lower_gray, upper_gray)

        # # 회색 영역을 흰색(255)으로 설정
        # resized_image[mask > 0] = 255

        # 출력 경로 설정
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # result_image = adaptive_thresh
        result_image = binary_image
        
        image_name = image_name.split(".")[0]
        new_image_name = f"{image_name}(processed).png"
        
        output_path = os.path.join(output_dir, new_image_name)
        cv2.imwrite(output_path, result_image)
        print(f"Processed image saved at {output_path}")
        
        return new_image_name
    
    except Exception as e:
        print(f"An error occurred: {e}")
    
    return None