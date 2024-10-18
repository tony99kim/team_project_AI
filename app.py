from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import requests
from io import BytesIO
import firebase_admin
from firebase_admin import credentials, firestore, storage
import torch
from torchvision import transforms

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

# BLIP 모델 및 프로세서 로드
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

# 신뢰도 기준 설정
CONFIDENCE_THRESHOLD = 0.7  # 70% 신뢰도 이상일 때만 승인

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

            # 1차: 설명과 이미지의 관계 판단
            description_image_question = f"Is the image related to the description '{description}'?"
            inputs = processor(images=img_tensor, text=description_image_question, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            image_related = logits.argmax().item()  # 1이면 관련 있음
            confidence = torch.softmax(logits, dim=1).max().item()  # 확률 값

            # 디버깅 로그 추가
            print(f"Image URL: {image_url}, Image Related: {image_related}, Confidence: {confidence}")

            # 1차 결과 판단
            if confidence < CONFIDENCE_THRESHOLD or image_related == 0:  # 신뢰도 기준 미달 또는 관련 없음
                final_status = '승인 거부 1'
                break  # 1차에서 거부되면 반복 종료

            # 2차: 제목과 설명의 관계 판단
            title_description_question = f"Is the description '{description}' related to the title '{title}'?"
            inputs = processor(images=img_tensor, text=title_description_question, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            title_related = logits.argmax().item()  # 1이면 관련 있음
            confidence = torch.softmax(logits, dim=1).max().item()  # 확률 값

            # 디버깅 로그 추가
            print(f"Title Related: {title_related}, Confidence: {confidence}")

            # 2차 결과 판단
            if confidence < CONFIDENCE_THRESHOLD or title_related == 0:  # 신뢰도 기준 미달 또는 관련 없음
                final_status = '승인 거부 2'
                break  # 2차에서 거부되면 반복 종료

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

                # 1차: 설명과 이미지의 관계 판단
                description_image_question = f"Is the image related to the description '{description}'?"
                inputs = processor(images=img_tensor, text=description_image_question, return_tensors="pt")

                with torch.no_grad():
                    outputs = model(**inputs)
                logits = outputs.logits
                image_related = logits.argmax().item()
                confidence = torch.softmax(logits, dim=1).max().item()  # 확률 값

                # 디버깅 로그 추가
                print(f"Image URL: {image_url}, Image Related: {image_related}, Confidence: {confidence}")

                # 1차 결과 판단
                if confidence < CONFIDENCE_THRESHOLD or image_related == 0:  # 관련이 없거나 신뢰도 미달
                    final_status = '승인 거부 1'
                    break  # 1차에서 거부되면 반복 종료

                # 2차: 제목과 설명의 관계 판단
                title_description_question = f"Is the description '{description}' related to the title '{title}'?"
                inputs = processor(images=img_tensor, text=title_description_question, return_tensors="pt")

                with torch.no_grad():
                    outputs = model(**inputs)
                logits = outputs.logits
                title_related = logits.argmax().item()
                confidence = torch.softmax(logits, dim=1).max().item()  # 확률 값

                # 디버깅 로그 추가
                print(f"Title Related: {title_related}, Confidence: {confidence}")

                # 2차 결과 판단
                if confidence < CONFIDENCE_THRESHOLD or title_related == 0:  # 관련이 없거나 신뢰도 미달
                    final_status = '승인 거부 2'
                    break  # 2차에서 거부되면 반복 종료

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

def preprocess_image(image):
    """이미지를 모델 입력 형식으로 변환하는 함수"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)  # 배치 차원 추가

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # 서버 실행
