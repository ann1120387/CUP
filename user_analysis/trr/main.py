import torch
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset
from train_trr import train
import sys
import os

class FilteredImageFolder(ImageFolder):
    def __init__(self, root, transform=None, exclude_class=None):
        super().__init__(root, transform=transform)
        self.exclude_class = exclude_class
        
        # Filter out samples belonging to the excluded class
        if self.exclude_class:
            self.samples = [sample for sample in self.samples if self.exclude_class not in sample[0]]
            self.targets = [self.class_to_idx[os.path.basename(os.path.dirname(path))] for path, _ in self.samples]

if len(sys.argv) > 1:
    excluded_class = sys.argv[1]
    print(f"Excluding class: {excluded_class}")
else:
    excluded_class = None
    print("No class specified for exclusion.")

num_classes = 200

model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
ckpt_prefix = f"resnet18_{excluded_class}"

dataset_dirs = [
    r"datasets\clean_standard",
    r"datasets\wet_standard",
    r"datasets\warm_standard",
    r"datasets\dirty_standard",
    r"datasets\clean_notstandard",
    r"datasets\wet_notstandard",
    r"datasets\warm_notstandard",
    r"datasets\dirty_notstandard"
]

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets while excluding the specified class
train_datasets = [FilteredImageFolder(root=f'{dir_path}/train', transform=transform, exclude_class=excluded_class) for dir_path in dataset_dirs]
combined_train_dataset = ConcatDataset(train_datasets)
num_train_samples = len(combined_train_dataset)
subset_size = num_train_samples // 8
indices = torch.randperm(num_train_samples)[:subset_size]  
train_dataset = Subset(combined_train_dataset, indices)  

val_datasets = [FilteredImageFolder(root=f'{dir_path}/val', transform=transform, exclude_class=excluded_class) for dir_path in dataset_dirs]
combined_val_dataset = ConcatDataset(val_datasets)
num_val_samples = len(combined_val_dataset)
subset_size = num_val_samples // 8
indices = torch.randperm(num_val_samples)[:subset_size]  
val_dataset = Subset(combined_val_dataset, indices)  

test_datasets = [FilteredImageFolder(root=f'{dir_path}/test', transform=transform, exclude_class=excluded_class) for dir_path in dataset_dirs]
combined_test_dataset = ConcatDataset(test_datasets)
num_test_samples = len(combined_test_dataset)
subset_size = num_test_samples // 8
indices = torch.randperm(num_test_samples)[:subset_size]  
test_dataset = Subset(combined_test_dataset, indices)  

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
num_epochs = 6

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(device)

train(ckpt_prefix, model, device, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=num_epochs)

