import os
import sys
import json
import torch
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset
from util import prep_dataset, combine_dirs
from train import train
from evaluate import evaluate_model
from torchvision.models import vit_b_16, ViT_B_16_Weights

def initialize_transform(transform_config):
    return transforms.Compose([
        transforms.Resize(transform_config["resize"]),
        transforms.CenterCrop(transform_config["crop"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=transform_config["normalize_mean"], std=transform_config["normalize_std"])
    ])

def initialize_model(setting, num_classes):
    if setting["model"] == "vit_b_16":
        model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        in_features = model.heads.head.in_features
        model.heads = torch.nn.Sequential(torch.nn.Linear(in_features=in_features, out_features=num_classes))
    elif setting["model"] == "deit_b_16":
        model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
        in_features = 768
        model.heads = torch.nn.Sequential(torch.nn.Linear(in_features=in_features, out_features=num_classes))
    elif setting["model"] == "resnet18":
        model = models.resnet18(pretrained=True)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
    elif setting["model"] == "mobilenet":
        model = models.mobilenet_v3_small(pretrained=True)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    elif setting["model"] == "densenet":
        model = models.densenet161(pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = torch.nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f"Unsupported model: {setting['model']}")

    if not setting.get("train", True) and "checkpoint_path" in setting and setting["checkpoint_path"]:
        print(f"Loading model checkpoint from {setting['checkpoint_path']}")
        checkpoint = torch.load(setting["checkpoint_path"], map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
    return model

def create_dataset(dataset_dirs, subset_ratio, transform):
    tmp_datasets = [ImageFolder(root=dir_path, transform=transform) for dir_path in dataset_dirs]
    combined_tmp_dataset = ConcatDataset(tmp_datasets)
    if subset_ratio < 1:
        num_train_samples = len(combined_tmp_dataset)
        subset_size = int(num_train_samples * subset_ratio)
        indices = torch.randperm(num_train_samples)[:subset_size]
        tmp_dataset = Subset(combined_tmp_dataset, indices)
    else:
        tmp_dataset = combined_tmp_dataset
    return tmp_dataset

def main(config_path):
    with open(config_path, "r") as file:
        config = json.load(file)
    
    num_classes = config["num_classes"]
    transform = initialize_transform(config["transform"])
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    num_epochs = config["num_epochs"]
    step_size = config["scheduler"]["step_size"]
    gamma = config["scheduler"]["gamma"]

    for setting in config["settings"]:
        print(f"Processing setting: {setting['ckpt_prefix']}")
        
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        
        model = initialize_model(setting, num_classes)
        train_dataset = create_dataset([os.path.join("datasets", dir_path, "train") for dir_path in setting["dataset_dirs"]], setting["subset_ratio"], transform)
        val_dataset = create_dataset([os.path.join("datasets", dir_path, "val") for dir_path in setting["dataset_dirs"]], setting["subset_ratio"], transform)
        test_dataset = create_dataset([os.path.join("datasets", dir_path, "test") for dir_path in setting["dataset_dirs"]], setting["subset_ratio"], transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        if setting.get("train", True):  
            print("Training the model...")
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

            model = model.to(device)
            print(f"Using device: {device}")

            train(setting['ckpt_prefix'], model, device, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=num_epochs)
        else:
            print("Skipping training, evaluating the model...")
        
        evaluate_model(model, test_loader, device, setting['ckpt_prefix'])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <config.json>")
        sys.exit(1)

    config_path = sys.argv[1]
    main(config_path)
