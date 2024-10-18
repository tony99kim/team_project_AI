import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import storage
from datetime import datetime
import requests  # requests 라이브러리 추가

# Firebase Admin SDK 인증
cred = credentials.Certificate('C:/Users/taeyeop/team_project_AI/firebase_service_account.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'team-project-12345.appspot.com'  # 실제 버킷 이름으로 변경
})

# Firestore 클라이언트 및 Storage 버킷 생성
db = firestore.client()
bucket = storage.bucket()

def get_authentication_data():
    """Firestore에서 인증 데이터 가져오기 함수"""
    docs = db.collection('pointAuthentications').stream()
    return [doc.to_dict() for doc in docs]

def get_image_urls(authentication_id):
    """해당 인증 ID에 대한 이미지 URL 가져오기"""
    image_urls = []

    # Storage에서 해당 인증 ID의 이미지 경로를 가져와서 URL 생성
    blobs = bucket.list_blobs(prefix=f'PointAuthenticationImages/{authentication_id}/')
    for blob in blobs:
        image_urls.append(blob.public_url)  # 공개 URL 추가

    return image_urls

def analyze_data(title, description, images):
    """AI 모델을 통한 데이터 분석 (API 호출)"""
    response = requests.post('http://127.0.0.1:5000/api/evaluate', json={
        'title': title,
        'description': description,
        'images': images
    })

    if response.status_code == 200:
        return response.json().get('status')
    else:
        print('Error calling AI model:', response.status_code, response.text)
        return None  # API 호출 실패 시 None 반환

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
    main()
