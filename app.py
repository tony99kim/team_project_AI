from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
from io import BytesIO
import firebase_admin
from firebase_admin import credentials, firestore, storage
import torch
from torchvision import transforms
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import os

app = Flask(__name__)

# CORS 설정
CORS(app)

# Firebase Admin SDK 인증
cred = credentials.Certificate('C:/Users/taeyeop/team_project_AI/firebase_service_account.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'team-project-12345.appspot.com'
})

db = firestore.client()
bucket = storage.bucket()

# 모델 경로 설정
MODEL_DIR = 'path/to/save/model'

# 모델 및 프로세서 로드 함수
def load_model():
    processor = BlipProcessor.from_pretrained(MODEL_DIR)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_DIR)
    return processor, model

# 초기 모델 로드
processor, model = load_model()

# 신뢰도 기준 설정
CONFIDENCE_THRESHOLD = 0.7  # 70% 신뢰도 이상일 때만 승인

def preprocess_image(image):
    """이미지를 모델 입력 형식으로 변환하는 함수"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)  # 배치 차원 추가

def generate_caption(image):
    """이미지에서 캡션 생성"""
    inputs = processor(images=image, return_tensors="pt")
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

def is_description_similar(caption, description):
    """캡션과 설명의 의미적 유사도 판단"""
    vectorizer = CountVectorizer().fit_transform([caption, description])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)
    return cosine_sim[0][1] > 0.7  # 유사도가 0.7 이상이면 일치한다고 판단

@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    data = request.json
    title = data['title']
    description = data['description']
    image_urls = data['images']
    
    # Firestore 문서 참조
    doc_ref = db.collection('pointAuthentications').document(title)  # title을 ID로 사용

    final_status = None  # 상태 초기화

    for image_url in image_urls:
        # 이미지를 요청하여 가져오기
        response = requests.get(image_url)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img_tensor = preprocess_image(img)

            # 이미지에서 캡션 생성
            caption = generate_caption(img)
            print(f"Generated Caption: {caption}")

            # 1차: 설명과 이미지의 의미적 유사도 판단
            if not is_description_similar(caption, description):
                final_status = '승인 거부: 설명 불일치'
                break  # 설명 불일치 시 반복 종료

            # 2차: 제목과 설명의 관계 판단
            if not is_description_similar(description, title):
                final_status = '승인 거부: 제목과 설명 불일치'
                break  # 제목과 설명 불일치 시 반복 종료

        else:
            return jsonify({'status': '오류', 'error': f'이미지 가져오기 실패: {image_url}'})

    # 최종 상태 결정
    if final_status is None:  # 거부되지 않은 경우
        final_status = '승인'  # 모든 판단이 통과하면 승인

    # Firestore에 상태 업데이트
    doc_ref.update({'status': final_status})  # 기존 데이터 유지하며 상태 업데이트

    return jsonify({'status': final_status})  # 최종 상태 반환

@app.route('/api/autoApprove', methods=['POST'])
def auto_approve():
    certifications = db.collection('pointAuthentications').where('status', '==', '대기').stream()
    results = []

    for cert in certifications:
        data = cert.to_dict()
        title = data['title']
        description = data['description']
        authentication_id = data['authenticationId']

        # Storage에서 이미지 URL 가져오기
        image_urls = get_image_urls(authentication_id)

        if not image_urls:  # 이미지가 없으면 처리하지 않음
            final_status = '이미지 없음'
            results.append({'id': cert.id, 'status': final_status})
            continue

        # Firestore 문서 참조
        doc_ref = db.collection('pointAuthentications').document(cert.id)  # ID로 사용
        final_status = None  # 상태 초기화

        for image_url in image_urls:
            response = requests.get(image_url)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content)).convert("RGB")
                img_tensor = preprocess_image(img)

                # 이미지에서 캡션 생성
                caption = generate_caption(img)

                # 1차: 설명과 캡션의 의미적 유사도 판단
                if not is_description_similar(caption, description):
                    final_status = '승인 거부: 설명 불일치'
                    break  # 설명 불일치 시 반복 종료

                # 2차: 제목과 설명의 관계 판단
                if not is_description_similar(description, title):
                    final_status = '승인 거부: 제목과 설명 불일치'
                    break  # 제목과 설명 불일치 시 반복 종료

            else:
                results.append({'id': cert.id, 'status': '오류', 'error': f'이미지 가져오기 실패: {image_url}'})
                continue

        # Firestore에 상태 업데이트
        if final_status is None:  # 모든 판단이 통과한 경우에만 승인
            final_status = '승인'

        doc_ref.update({'status': final_status})  # 기존 데이터 유지하며 상태 업데이트

        results.append({'id': cert.id, 'status': final_status})

    return jsonify({'results': results})

def get_image_urls(authentication_id):
    """해당 인증 ID에 대한 이미지 URL 가져오기"""
    images_ref = bucket.list_blobs(prefix=f'PointAuthenticationImages/{authentication_id}/')
    image_urls = []

    for blob in images_ref:
        image_urls.append(blob.public_url)  # 공개 URL 추가

    return image_urls

def get_authentication_data():
    """Firestore에서 인증 데이터 가져오기 함수"""
    docs = db.collection('pointAuthentications').stream()
    return [doc.to_dict() for doc in docs]

def update_authentication_status(authentication_id, status):
    """인증 상태 업데이트 함수"""
    db.collection('pointAuthentications').document(authentication_id).update({
        'status': status,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    print(f'Authentication ID {authentication_id} status updated to {status}')

def main():
    # 인증 데이터 가져오기
    authentication_data = get_authentication_data()
    
    for data in authentication_data:
        authentication_id = data.get('authenticationId')  # 인증 ID 가져오기
        if not authentication_id:
            print('No authentication ID found for document:', data)
            continue  # 인증 ID가 없으면 다음 문서로 넘어감

        title = data.get('title', '')
        description = data.get('description', '')

        # 이미지 URL 가져오기
        image_urls = get_image_urls(authentication_id)

        # 이미지가 없으면 상태 업데이트 하지 않음
        if not image_urls:
            print(f'No images found for Authentication ID {authentication_id}. Skipping...')
            continue

        # AI 모델 분석 (API 호출)
        status = analyze_data(title, description, image_urls)

        # API 호출이 성공적으로 이루어진 경우에만 상태 업데이트
        if status:
            update_authentication_status(authentication_id, status)
        else:
            print(f'Failed to analyze data for Authentication ID {authentication_id}. Skipping status update.')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # 서버 실행
