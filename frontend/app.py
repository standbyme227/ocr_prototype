import requests
import streamlit as st
import os


# UI 구성
st.title("🖼️ Preprocessing and 🔍 OCR")

# 기본 저장 경로 설정
original_folder = "/volumes/original"

# Streamlit UI
st.header("1. File Upload to Original Folder")

# 파일 업로드 위젯 (여러 파일 선택 가능)
uploaded_files = st.file_uploader(
    "Choose image files to upload", accept_multiple_files=True, type=["png", "jpg", "jpeg"]
)

# 사용자 입력: 저장할 폴더 이름
target_folder = st.text_input("Specify a folder name:", placeholder="Enter folder name")

if st.button("Upload Files"):
    if not target_folder:
        st.error("Please specify a folder name.")
    elif not uploaded_files:
        st.error("Please select at least one file.")
    else:
        # 저장 경로 생성
        target_path = os.path.join(original_folder, target_folder)
        if not os.path.exists(target_path):
            os.makedirs(target_path, exist_ok=True)

        # 파일 저장
        for uploaded_file in uploaded_files:
            file_path = os.path.join(target_path, uploaded_file.name)
            if os.path.exists(file_path):
                st.warning(f"File '{uploaded_file.name}' already exists and will be skipped.")
            else:
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

        st.success(f"Uploaded {len(uploaded_files)} files to folder '{target_folder}'")

# 2단계: 전처리
st.header("2. Preprocess Images")
original_folders = [d for d in os.listdir("/volumes/original") if os.path.isdir(os.path.join("/volumes/original", d))]
selected_original_folder = st.selectbox("Select a folder from original:", [""] + original_folders)

if selected_original_folder:
    preprocess_button = st.button("Run Preprocessing")
    if preprocess_button:
        response = requests.post(
            "http://preprocessor:5001/preprocess",
            json={"input_folder": os.path.join("/volumes/original", selected_original_folder)}
        )
        if response.status_code == 200:
            st.success(f"Preprocessed folder saved at: {response.json()['output_folder']}")
        else:
            st.error(f"Preprocessing failed: {response.text}")

# 3단계: OCR
st.header("3. Perform OCR")
preprocessed_folders = [d for d in os.listdir("/volumes/preprocessed") if os.path.isdir(os.path.join("/volumes/preprocessed", d))]
selected_preprocessed_folder = st.selectbox("Select a folder from preprocessed:", [""] + preprocessed_folders)

if selected_preprocessed_folder:
    ocr_button = st.button("Run OCR")
    if ocr_button:
        response = requests.post(
            "http://ocr_server:8082/ocr",
            json={"folder_path": os.path.join("/volumes/preprocessed", selected_preprocessed_folder)}
        )
        if response.status_code == 200:
            st.success(f"OCR complete. Results saved at: {response.json()['output_folder']}")
        else:
            st.error(f"OCR failed: {response.text}")