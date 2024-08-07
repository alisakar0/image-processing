import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import numpy as np
import os

# GPU cihazını ayarla
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet50(pretrained=True).to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_and_transform_images(image_paths, transform, device):
    images = []
    for image_path in image_paths:
        try:
            image = Image.open(image_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)
            images.append(image)
        except Exception as e:
            print(f"Görüntü yükleme ve dönüştürme hatası: {e}")
    return images

def extract_features_batch(images, model, device):
    images_tensor = torch.cat(images)  # Tüm resimleri tek bir tensor olarak birleştir
    with torch.no_grad():
        features = model(images_tensor)  # Modeli tüm resim tensorü üzerinde çalıştır
    return features

image_folder_path = '/content/drive/MyDrive/Colab Notebooks/products'
image_paths = [os.path.join(image_folder_path, file_name) for file_name in os.listdir(image_folder_path) if file_name.endswith(('jpg', 'jpeg', 'png'))]

batch_size = 64  # Bellek sınırlarını göz önünde bulundurarak bir batch boyutu seçin
all_features = []
all_image_paths = []

for i in range(0, len(image_paths), batch_size):
    batch_paths = image_paths[i:i + batch_size]
    images = load_and_transform_images(batch_paths, transform, device)
    if images:
        features = extract_features_batch(images, model, device)
        features_np = features.squeeze().cpu().numpy()  # GPU'dan CPU'ya taşı ve numpy dizisine dönüştür
        all_features.append(features_np)
        all_image_paths.extend(batch_paths)

if all_features:
    feature_matrix = np.concatenate(all_features, axis=0)
    np.save('features.npy', feature_matrix)
    np.save('image_paths.npy', all_image_paths)

    print("Özellikler başarıyla kaydedildi.")
else:
    print("Hiçbir özellik çıkarılamadı.")