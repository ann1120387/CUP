import os
import random
import pickle
import torch
from PIL import Image
from torch.autograd import Variable
import pickle
from PIL import Image
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, auc
import matplotlib.pyplot as plt
import pickle
import random
from torchvision import models, transforms

def get_random_images(directory):
    try:
        all_files = os.listdir(directory)
        image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))]
        selected_image_paths = [os.path.join(directory, img) for img in image_files]
    except Exception as err:
        return None
    return selected_image_paths

def get_image_with_smallest_number(directory):
    files = os.listdir(directory)
    png_files = [f for f in files if f.endswith('.png')]
    png_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    return os.path.join(directory, png_files[0]) if png_files else None

def find_pth_files(model_root, user_id):
    for root, dirs, files in os.walk(model_root):
        for file in files:
            if file.endswith(".pth") and f"_{user_id}_" in file:
                return os.path.join(root, file)
    return 0

def extract_feature_vector_cnn(model, image_path, transform, layer, device, emb_dict, emb_size):
    if image_path in emb_dict.keys():
        return emb_dict[image_path], emb_dict

    image = Image.open(image_path).convert('RGB')
    
    t_img = Variable(transform(image)).unsqueeze(0)
    t_img = t_img.to(device)

    model.eval()
    my_embedding = torch.zeros(1, emb_size, 1, 1)

    def copy_data(m, i, o):
        try:
            my_embedding.copy_(o.data)
        except Exception as err:
            # print(str(err))
            pooled_output = torch.nn.AdaptiveAvgPool2d((1, 1))(o)
            my_embedding.copy_(pooled_output.data)

    h = layer.register_forward_hook(copy_data)
    model(t_img)
    h.remove()
    features = my_embedding.squeeze()
    emb_dict[image_path] = features

    return features, emb_dict

def get_scores_cnn(prefix, model, transform, layer, device, pairs, emb_size):
    scores = []
    labels = []

    emb_dir = r".\embeddings"
    os.makedirs(emb_dir, exist_ok=True)

    emb_path = rf'{emb_dir}\{prefix}_emb_verification.pkl'
    if not os.path.exists(emb_path):
        with open(emb_path, 'wb') as f:
            emb_dict = {}
            pickle.dump(emb_dict, f)
    
    with open(emb_path, 'rb') as f:
        emb_dict = pickle.load(f)

    # the scores are similarity scores, however for roc curve, we take diff as positive class
    # thus, to make the socres probability estimates of the positive class
    # use (1 - similarity) as the final scores
    for [img1, img2] in pairs:
        features1, emb_dict = extract_feature_vector_cnn(model, img1, transform, layer, device, emb_dict, emb_size)
        features2, emb_dict = extract_feature_vector_cnn(model, img2, transform, layer, device, emb_dict, emb_size)

        cosine_similarity = torch.nn.functional.cosine_similarity(features1, features2, dim=0).item()

        scores.append(1-cosine_similarity)
        labels.append(0)

    with open(emb_path, 'wb') as file:
        pickle.dump(emb_dict, file)

    return scores, labels

def create_pairs(user, num_pairs):
    users = [f"{i:03}_L" for i in range(1, 101)]
    users.remove(user)
    profiles_dir = r"..\..\train_eval\datasets\clean_standard\test"
    profiles = []
    for other_user in users:
        profiles.append(get_image_with_smallest_number(rf"{profiles_dir}\{other_user}"))

    cur_images = []
    for condition in ["clean", "wet", "dirty", "warm"]:
        for pose in ["notstandard", "standard"]:
            root = r"..\..\train_eval\datasets"
            tmp_dir = rf"{root}\{condition}_{pose}\test\{user}"
            if get_random_images(tmp_dir):
                cur_images+=get_random_images(tmp_dir)
    selected_images = random.sample(cur_images, num_pairs)
    pairs = []
    for img in selected_images:
        for profile in profiles:
            pairs.append((img, profile))
    return pairs

# use the left hand images of each user
users = [f"{i:03}_L" for i in range(1, 101)]
model_root = r"res"

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

for user in users:
    pairs = create_pairs(user, 5)
    model_path = find_pth_files(model_root, user)
    model = models.resnet18(pretrained=True)
    layer = model._modules.get('avgpool')
    prefix = "resnet18"
    num_classes = 200
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    emb_size = 512
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    scores, labels = get_scores_cnn(prefix, model, transform, layer, device, pairs, emb_size)
    file_path = os.path.join(r"scores", f"{user}.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(scores, f)