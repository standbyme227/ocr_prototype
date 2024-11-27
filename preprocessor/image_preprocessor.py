import cv2
import numpy as np
import os

def remove_non_black_white_to_yellow(image):
    """
    이미지에서 흰색과 검은색을 제외한 색상을 노란색으로 변경.
    """
    # # 이미지가 BGR 컬러인지 확인
    # if len(image.shape) != 3:
    #     raise ValueError("Input image must be a color image (BGR).")

    # BGR 색상 범위에서 흰색(255, 255, 255)과 검은색(0, 0, 0)을 제외한 픽셀 변경
    result_image = image.copy()

    # 조건: 흰색 및 검은색 범위 마스크
    black_min = 0
    black_max = 60
    
    white_min = 180
    white_max = 255
    non_black_white_mask = ~(
        (
            (image[:, :, 0] >= black_min) & (image[:, :, 0] <= black_max) &  # B 채널 검은색 범위
            (image[:, :, 1] >= black_min) & (image[:, :, 1] <= black_max) &  # G 채널 검은색 범위
            (image[:, :, 2] >= black_min) & (image[:, :, 2] <= black_max)    # R 채널 검은색 범위
        ) |
        (
            (image[:, :, 0] >= white_min) & (image[:, :, 0] <= white_max) &  # B 채널 흰색 범위
            (image[:, :, 1] >= white_min) & (image[:, :, 1] <= white_max) &  # G 채널 흰색 범위
            (image[:, :, 2] >= white_min) & (image[:, :, 2] <= white_max)    # R 채널 흰색 범위
        )
    )

    # 노란색 BGR 값 설정
    yellow_color = [0, 255, 255]

    # 노란색으로 변경
    result_image[non_black_white_mask] = yellow_color

    return result_image

def enhance_to_white(image):
    """
    흑백 이미지에서 애매한 흰색(밝은 색)을 완전 흰색으로 변경.
    """
    # 흰색 범위 설정 (밝은 색 영역)
    lower_bound = 230  # 밝은 색 하한값
    upper_bound = 254  # 밝은 색 상한값

    # 마스크 생성 (밝은 색 영역 찾기)
    mask = cv2.inRange(image, lower_bound, upper_bound)

    # 밝은 색 영역을 완전 흰색으로 설정
    image[mask > 0] = 255

    return image

def enhance_gray_to_white(image):
    """
    컬러 이미지에서 회색 계통만 흰색으로 변경.
    """
    # 밝기 범위 설정 (회색의 밝기)
    lower_brightness = 100  # 밝기 하한값
    upper_brightness = 250  # 밝기 상한값

    # RGB 값의 차이가 작은 픽셀(회색 계통)을 찾는 조건
    diff_threshold = 5  # R, G, B 간 차이 허용 범위

    # R, G, B 채널 분리
    b, g, r = cv2.split(image)

    # 밝기 조건: 픽셀이 밝기 범위 내에 있는지 확인
    brightness_mask = (b >= lower_brightness) & (b <= upper_brightness) & \
                      (g >= lower_brightness) & (g <= upper_brightness) & \
                      (r >= lower_brightness) & (r <= upper_brightness)

    # R, G, B 값 차이가 diff_threshold 이하인지 확인
    gray_mask = (np.abs(r - g) <= diff_threshold) & \
                (np.abs(r - b) <= diff_threshold) & \
                (np.abs(g - b) <= diff_threshold)

    # 최종 마스크 생성: 밝기와 회색 조건 모두 만족
    mask = brightness_mask & gray_mask

    # 회색 계통 픽셀을 흰색으로 변경
    result_image = image.copy()
    result_image[mask] = [255, 255, 255]

    return result_image


# 전처리 함수
def preprocess_image(image_path, output_dir):
    try:
        # 이미지 읽기 (흑백으로 로드)
        # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(image_path)
        image_name = image_path.split("/")[-1]

        if image is None:
            print(f"Error: Could not load image from path {image_path}")
            return

        # Step 1: 해상도 업스케일링
        result_image = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_LANCZOS4)
        
        result_image = enhance_gray_to_white(result_image)
        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)
        # Step 2-2: 이진화
        _, result_image = cv2.threshold(result_image, 170, 255, cv2.THRESH_BINARY)
        
        # # Step 2-1: 잡음 제거 (선택 사항)
        # blurred_image = cv2.GaussianBlur(resized_image, (3, 3), 0)
        
        # # # Step 5: 문자 영역 강조 (선택 사항)
        # kernel = np.ones((1, 1), np.uint8)
        # result_image = cv2.erode(result_image, kernel, iterations=1)
        
        # result_image, _ = cv2.findContours(result_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # result_image = cv2.drawContours(image, result_image, -1, (255, 255, 0), 2)  # 노란색 컨투어



        # # Step 3: 어댑티브 이진화 (선택 사항, 기존 이진화 대체 가능)
        # result_image = cv2.adaptiveThreshold(
        #     result_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 0.3
        # )
    
        # # Step 4: 엣지 강조 (선택 사항)
        # edges = cv2.Canny(binary_image, 100, 200)

        # 출력 경로 설정
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # result_image = adaptive_thresh
        # # Step 2-1: 잡음 제거 (선택 사항)
        # result_image = cv2.GaussianBlur(result_image, (3, 3), 0)
        
        # # Step 5: 문자 영역 강조 (선택 사항)
        # kernel = np.ones((1, 1), np.uint8)
        # result_image = cv2.erode(result_image, kernel, iterations=1)
        
        image_name = image_name.split(".")[0]
        new_image_name = f"{image_name}(processed).png"
        
        output_path = os.path.join(output_dir, new_image_name)
        cv2.imwrite(output_path, result_image)
        print(f"Processed image saved at {output_path}")
        
        return new_image_name
    
    except Exception as e:
        return e