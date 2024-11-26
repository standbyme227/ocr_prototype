import requests
import streamlit as st

# UI 구성
st.title("Image Preprocessing and OCR")
input_folder = st.text_input("Input Folder Path", placeholder="Enter folder path here")
st.write("")

if input_folder:
    # 이미지 파일 수 확인
    st.write(f"Number of images in folder: {len(os.listdir(input_folder))}")

    # 전처리 버튼
    if st.button("Run Preprocessing"):
        response = requests.post("http://preprocessor:5001/preprocess", json={"input_folder": input_folder})
        st.success(response.json()["message"])

    # OCR 버튼
    if st.button("Run OCR"):
        response = requests.post("http://ocr_server:5002/ocr", json={"input_folder": input_folder})
        st.success(response.json()["message"])

    # 전체 작업 버튼
    if st.button("Run All"):
        preprocess_response = requests.post("http://preprocessor:5001/preprocess", json={"input_folder": input_folder})
        ocr_response = requests.post("http://ocr_server:5002/ocr", json={"input_folder": preprocess_response.json()["output_folder"]})
        st.success(f"OCR complete. Results saved at {ocr_response.json()['output_path']}")