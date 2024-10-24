import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BlipProcessor, BlipForConditionalGeneration
from torchvision import transforms
from PIL import Image
import firebase_admin
from firebase_admin import credentials, firestore, storage
import logging
import os

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# Firebase Admin SDK 인증
cred = credentials.Certificate('C:/Users/taeyeop/team_project_AI/firebase_service_account.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'team-project-12345.appspot.com' 
})

# Firestore 클라이언트 및 Storage 버킷 생성
db = firestore.client()
bucket = storage.bucket()

# 데이터셋 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, image_dir, captions):
        self.image_dir = image_dir
        self.captions = captions
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.captions[idx]['image'])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        caption = self.captions[idx]['caption']
        return image, caption

def get_authentication_data():
    """Firestore에서 인증 데이터 가져오기 함수"""
    approved_data = []
    denied_description_mismatch_data = []
    denied_title_mismatch_data = []
    
    # 승인된 인증글 가져오기
    approved_docs = db.collection('pointAuthentications').where('status', '==', '승인').stream()
    for doc in approved_docs:
        approved_data.append(doc.to_dict())
    
    # 설명 불일치로 거부된 인증글 가져오기
    denied_description_mismatch_docs = db.collection('pointAuthentications').where('status', '==', '승인 거부: 설명 불일치').stream()
    for doc in denied_description_mismatch_docs:
        denied_description_mismatch_data.append(doc.to_dict())

    # 제목 불일치로 거부된 인증글 가져오기
    denied_title_mismatch_docs = db.collection('pointAuthentications').where('status', '==', '승인 거부: 제목과 설명 불일치').stream()
    for doc in denied_title_mismatch_docs:
        denied_title_mismatch_data.append(doc.to_dict())
    
    return approved_data, denied_description_mismatch_data, denied_title_mismatch_data

def main():
    image_dir = 'path/to/images'  # 이미지 디렉토리
    approved_data, denied_description_mismatch_data, denied_title_mismatch_data = get_authentication_data()

    # 승인된 인증글과 거부된 인증글로부터 캡션 데이터 준비
    captions = []
    for data in approved_data:
        captions.append({'image': data['image'], 'caption': data['description']})
    for data in denied_description_mismatch_data:
        captions.append({'image': data['image'], 'caption': data['description']})
    for data in denied_title_mismatch_data:
        captions.append({'image': data['image'], 'caption': data['description']})

    dataset = CustomDataset(image_dir, captions)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 모델 및 프로세서 로드
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-vqa-base")

    # 훈련 설정
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # 훈련 루프
    model.train()
    for epoch in range(3):  # 에포크 수
        for batch in dataloader:
            images, captions = batch
            outputs = model(images, captions=processor(captions, return_tensors="pt", padding=True, truncation=True).input_ids)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logging.info(f'Epoch: {epoch}, Loss: {loss.item()}')

    # 모델 저장
    model.save_pretrained('path/to/save/model')
    processor.save_pretrained('path/to/save/model')

if __name__ == '__main__':
    main()
