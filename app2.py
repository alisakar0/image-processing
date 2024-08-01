import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import torch
import torchvision.transforms as transforms
from IPython.display import display, Image as IPImage
from google.colab import files

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

feature_matrix = np.load('features.npy')
image_paths = np.load('image_paths.npy', allow_pickle=True)

def find_similar_images(query_image_path, model, transform, feature_matrix, image_paths):
    query_features = extract_features(query_image_path, model, transform)
    if query_features is None:
        print(f"Sorgu resmi bulunamadı: {query_image_path}")
        return []

    similarities = cosine_similarity([query_features], feature_matrix)[0]
    similar_indices = similarities.argsort()[-10:][::-1]
    similar_images = [image_paths[i] for i in similar_indices]
    return similar_images

uploaded = files.upload()
query_image_path = list(uploaded.keys())[0]

similar_images = find_similar_images(query_image_path, model, transform, feature_matrix, image_paths)

print("Benzer resimler:")
for image_path in similar_images:
    try:
        display(IPImage(filename=image_path))
    except FileNotFoundError:
        print(f"Resim bulunamadı: {image_path}")
