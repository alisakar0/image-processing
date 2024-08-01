import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import numpy as np
import os

model = resnet50(pretrained=True)
model = model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image_path, model, transform):
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            features = model(image).numpy().flatten()
        return features
    except Exception as e:
        print(f"Özellik çıkarımı hatası: {e}")
        return None

image_folder_path = '/content/drive/MyDrive/Colab Notebooks/products'

image_paths = [os.path.join(image_folder_path, file_name) for file_name in os.listdir(image_folder_path) if file_name.endswith(('jpg', 'jpeg', 'png'))]

feature_list = []
for image_path in image_paths:
    features = extract_features(image_path, model, transform)
    if features is not None:
        feature_list.append(features)
    else:
        print(f"Hata: {image_path} için özellik çıkarımı yapılamadı.")

feature_matrix = np.array(feature_list)
np.save('features.npy', feature_matrix)
np.save('image_paths.npy', image_paths)

print("Özellikler başarıyla kaydedildi.")