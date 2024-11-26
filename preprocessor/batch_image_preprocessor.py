import os
from image_preprocessor import preprocess_image


def batch_processor(input_folder):
    """
    지정된 폴더의 이미지를 전처리하고 OCR 수행.
    결과 파일은 'results' 폴더에 저장.
    """
    # 입력 폴더 검증
    if not os.path.exists(input_folder):
        raise ValueError(f"Input folder '{input_folder}' does not exist.")

    # 실행 파일과 같은 계층에 'results' 폴더 생성
    base_output_folder = os.path.join(os.getcwd(), "results")
    if not os.path.exists(base_output_folder):
        os.makedirs(base_output_folder)
        print(f"Created results folder at {base_output_folder}")

    # 현재 input_folder 이름에 따라 별도 폴더 생성
    folder_name = os.path.basename(os.path.normpath(input_folder))
    output_folder = os.path.join(base_output_folder, folder_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    dir_list = os.listdir(input_folder)
    
    count = 0

    # 폴더의 이미지 파일 처리
    for filename in dir_list:
        file_path = os.path.join(input_folder, filename)

        # 이미지 파일 검증
        if not (filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))):
            print(f"Skipping non-image file: {filename}")
            continue

        # 전처리
        try:
            print(f"Preprocessing image: {file_path}")
            filename = preprocess_image(file_path, output_folder)
            
        except Exception as e:
            print(f"Error during preprocessing {filename}: {e}")
            continue
        
        if filename is None:
            print(f"Error during preprocessing {filename}")
            continue
        count += 1
        
    if count > 0:
        print(f"Processed {count} images of total {len(dir_list)} in folder.")
        return output_folder
    else:
        raise ValueError(f"No images found in folder: {input_folder}")


if __name__ == "__main__":
    # 하드코딩된 폴더 경로
    basic_path = "/Users/mini/not_work/project/ocr_leaflet/data/test_data/"
    folder_id = "Tesco_leaflet_23"
    
    input_folder = basic_path + folder_id
    
    try:
        process_folder(input_folder)
    except Exception as e:
        print(f"An error occurred: {e}")