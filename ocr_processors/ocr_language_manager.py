import paddle
from paddleocr import PaddleOCR

class OCRLanguageManager:
    def __init__(self):
        self.models = {}

    def get_ocr_model(self, language="sk"):
        """
        언어에 따라 PaddleOCR 모델을 반환.
        이미 초기화된 모델이 있다면 캐시에서 반환.
        """
        if language not in self.models:
            print(f"Initializing PaddleOCR model for language: {language}")
            
            # # PaddlePaddle 설정 조정
            # paddle.set_device('cpu')  # GPU 대신 CPU 사용
            # paddle.set_flags({'FLAGS_use_mkldnn': False})  # MKL-DNN 비활성화

            # PaddleOCR 초기화 시 추가 파라미터 설정
            self.models[language] = PaddleOCR(
                use_angle_cls=True,
                lang=language,
                use_gpu=False,
                enable_mkldnn=False
            )
        return self.models[language]