import os
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from dotenv import load_dotenv
import subprocess

# 현재 디렉토리 설정
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# env 파일 로드
load_dotenv()

# 환경 변수 설정
TRAIN_DATA_NAME = os.getenv("TRAIN_DATA_NAME")

# YOLO 결과 경로
results_csv = os.path.join(PROJECT_DIR, 'results', TRAIN_DATA_NAME, 'results.csv')
tensorboard_log_dir = os.path.join(PROJECT_DIR, 'results', TRAIN_DATA_NAME, 'tensorboard')


def check_tensorboard_logs():
    """
    TensorBoard 로그 폴더 확인 및 변환 여부 결정
    Returns:
        bool: 변환 여부
    """
    if os.path.exists(tensorboard_log_dir) and os.listdir(tensorboard_log_dir):
        user_input = input("TensorBoard 로그 폴더가 이미 존재합니다. 변환을 다시 하시겠습니까? (yes/no): ").lower()
        return user_input == "yes"
    return True


def convert_results_to_tensorboard():
    """
    results.csv 파일을 읽어 TensorBoard 형식으로 변환
    """
    if os.path.exists(results_csv):
        results = pd.read_csv(results_csv)
        print("Results CSV loaded.")

        # TensorBoard SummaryWriter 초기화
        writer = SummaryWriter(log_dir=tensorboard_log_dir)

        # TensorBoard 로그 작성
        for epoch, row in results.iterrows():
            writer.add_scalar("train/box_loss", row["train/box_loss"], epoch)
            writer.add_scalar("train/cls_loss", row["train/cls_loss"], epoch)
            writer.add_scalar("train/dfl_loss", row["train/dfl_loss"], epoch)
            writer.add_scalar("metrics/precision(B)", row["metrics/precision(B)"], epoch)
            writer.add_scalar("metrics/recall(B)", row["metrics/recall(B)"], epoch)
            writer.add_scalar("metrics/mAP50(B)", row["metrics/mAP50(B)"], epoch)
            writer.add_scalar("metrics/mAP50-95(B)", row["metrics/mAP50-95(B)"], epoch)
            writer.add_scalar("val/box_loss", row["val/box_loss"], epoch)
            writer.add_scalar("val/cls_loss", row["val/cls_loss"], epoch)
            writer.add_scalar("val/dfl_loss", row["val/dfl_loss"], epoch)
            writer.add_scalar("learning_rate/pg0", row["lr/pg0"], epoch)
            writer.add_scalar("learning_rate/pg1", row["lr/pg1"], epoch)
            writer.add_scalar("learning_rate/pg2", row["lr/pg2"], epoch)

        print("TensorBoard logs written.")
        writer.close()
    else:
        print("Results CSV not found. 변환을 중단합니다.")
        exit()


def run_tensorboard():
    """
    TensorBoard 실행
    """
    user_input = input("TensorBoard를 실행하시겠습니까? (yes/no): ").lower()
    if user_input == "yes":
        print("TensorBoard 실행 중...")
        tensorboard_command = f"tensorboard --logdir={tensorboard_log_dir} --host=0.0.0.0 --port=6006"
        try:
            subprocess.run(tensorboard_command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"TensorBoard 실행 중 오류 발생: {e}")
    else:
        print("TensorBoard 실행이 취소되었습니다.")


if __name__ == "__main__":
    if check_tensorboard_logs():
        convert_results_to_tensorboard()
    run_tensorboard()