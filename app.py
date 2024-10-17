from flask import Flask, request, jsonify
from flask_cors import CORS  # CORS 라이브러리 import
from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO
import firebase_admin
from firebase_admin import credentials, firestore, storage

app = Flask(__name__)

# CORS 설정
CORS(app)  # 모든 도메인에서의 요청을 허용

# Firebase Admin SDK 인증
cred = credentials.Certificate('C:/Users/taeyeop/team_project_AI/firebase_service_account.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'team-project-12345.appspot.com'  # 실제 버킷 이름으로 변경
})

db = firestore.client()

# 모델 로드
nlp = pipeline("sentiment-analysis")
image_model = pipeline("image-classification")

@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    data = request.json
    title = data['title']
    description = data['description']
    image_urls = data['images']

    # 설명을 기반으로 감정 분석
    sentiment = nlp(f"Description: {description}")[0]

    # 이미지 분석
    title_matches = False

    for image_url in image_urls:
        # 이미지를 요청하여 가져오기
        response = requests.get(image_url)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            image_result = image_model(img)[0]
            if image_result['label'].lower() in title.lower().split():
                title_matches = True
        else:
            return jsonify({'status': '승인 거부', 'error': f'Failed to retrieve image: {image_url}'})

    # 결과 판단
    if sentiment['label'] == 'POSITIVE' and title_matches:
        return jsonify({'status': '승인'})
    else:
        return jsonify({'status': '승인 거부'})

@app.route('/api/autoApprove', methods=['POST'])
def auto_approve():
    certifications = request.json['certifications']
    results = []

    for cert in certifications:
        title = cert['title']
        description = cert['description']
        image_urls = cert['images']

        # 설명과 이미지를 분석하여 승인 여부 판단
        sentiment = nlp(f"Description: {description}")[0]
        title_matches = False

        for image_url in image_urls:
            response = requests.get(image_url)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                image_result = image_model(img)[0]
                if image_result['label'].lower() in title.lower().split():
                    title_matches = True
            else:
                return jsonify({'status': '승인 거부', 'error': f'Failed to retrieve image: {image_url}'})

        # 결과 저장
        if sentiment['label'] == 'POSITIVE' and title_matches:
            status = '승인'
        else:
            status = '승인 거부'

        # Firestore에 상태 업데이트
        doc_ref = db.collection('pointAuthentications').document(cert['id'])
        doc_ref.update({
            'status': status
        })

        results.append({'id': cert['id'], 'status': status})

    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # 서버 실행
