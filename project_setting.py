import requests
import os

# env 파일 로드
from dotenv import load_dotenv
load_dotenv()

# 환경 변수 설정
LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL")  # Label Studio 서버 URL
API_TOKEN = os.getenv("LABEL_STUDIO_API_TOKEN")

headers = {
    "Authorization": f"Token {API_TOKEN}",
    "Content-Type": "application/json"
}

def configure_project(project_id):
    # 1. label_config 설정
    default_label_config = """
    <View>
      <Image name="image" value="$image"/>
      <Choices name="label" toName="image" choice="single">
        <Choice value="Label1"/>
        <Choice value="Label2"/>
      </Choices>
    </View>
    """
    # 현재 label_config 확인
    response = requests.get(f"{LABEL_STUDIO_URL}/api/projects/{project_id}", headers=headers)
    if response.status_code == 200:
        current_label_config = response.json().get("label_config", "")
        if current_label_config != default_label_config:
            requests.patch(
                f"{LABEL_STUDIO_URL}/api/projects/{project_id}",
                headers=headers,
                json={"label_config": default_label_config}
            )
            print(f"프로젝트 {project_id}에 label_config 설정 완료!")
    else:
        print(f"label_config 조회 실패: {response.status_code} - {response.text}")

    # 2. ML Backend 설정
    ml_backend_url = f"{LABEL_STUDIO_URL}/api/projects/{project_id}/machine-learning"
    response = requests.get(ml_backend_url, headers=headers)
    if response.status_code == 200:
        ml_backends = response.json()
        if not ml_backends:  # ML Backend가 없으면 설정
            ml_backend_payload = {
                "name": "Default ML Backend",
                "model_url": "http://ml-backend:9090/predict",
                "protocol": "http"
            }
            requests.post(ml_backend_url, headers=headers, json=ml_backend_payload)
            print(f"프로젝트 {project_id}에 ML Backend 설정 완료!")
        else:
            print(f"프로젝트 {project_id}에는 이미 ML Backend가 설정되어 있습니다.")
    else:
        print(f"ML Backend 조회 실패: {response.status_code} - {response.text}")

    # 3. Webhook 설정
    webhook_url = f"{LABEL_STUDIO_URL}/api/projects/{project_id}/webhooks"
    response = requests.get(webhook_url, headers=headers)
    if response.status_code == 200:
        webhooks = response.json()
        if not webhooks:  # Webhook이 없으면 설정
            webhook_payload = {
                "url": "http://your-webhook-url.com/webhook",
                "event": "create_annotation",
                "headers": {
                    "Content-Type": "application/json"
                }
            }
            requests.post(webhook_url, headers=headers, json=webhook_payload)
            print(f"프로젝트 {project_id}에 Webhook 설정 완료!")
        else:
            print(f"프로젝트 {project_id}에는 이미 Webhook이 설정되어 있습니다.")
    else:
        print(f"Webhook 조회 실패: {response.status_code} - {response.text}")

def main():
    project_id = 1  # 새로 생성된 프로젝트 ID
    configure_project(project_id)

if __name__ == "__main__":
    main()