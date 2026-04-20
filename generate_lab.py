import json
import os

with open('LESSON_2B.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# We will just recreate the notebook from scratch for simplicity but keeping markdown and style
new_cells = []

def add_md(text):
    new_cells.append({
      "cell_type": "markdown",
      "metadata": {},
      "source": text.splitlines(keepends=True)
    })

def add_code(text):
    new_cells.append({
      "cell_type": "code",
      "metadata": {},
      "execution_count": None,
      "outputs": [],
      "source": text.splitlines(keepends=True)
    })

# Add title
add_md("<img src=\"https://drive.google.com/uc?export=view&id=1TFC0coLdLbK9Lf3_Ia2FDgw9AoqGf3bT\" width=180, align=\"center\"/>\n\nMaster's degree in Intelligent Systems\n\nSubject: 11754 - Deep Learning\n\nYear: 2025-2026\n\nProfessor: Miguel Ángel Calafat Torrens")
add_md("# LAB 2 - CNN and transfer learning")

add_code("""# Setup
import pathlib
import sys

PROJECT_DIR = str(pathlib.Path().resolve())
sys.path.append(PROJECT_DIR)
""")

add_code("""# Import libraries
import os
from torchvision import datasets, models
import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights
from torch.utils.data import random_split, Dataset
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import copy
from PIL import Image
import cv2
import helper_L2 as hp
from glob import glob
import random
""")

add_code("""# Global variables and setup
SEED = 0
IMG_FOLDER = PROJECT_DIR + "/weather_images"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VALID_SIZE = 0.2
TEST_SIZE = 0.2

print('Using {}'.format(DEVICE))
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
""")

add_code("""# Find all files. Images are now in a single folder without subfolders
image_paths = glob(IMG_FOLDER + "/*.jpg")
print('Number of images: {}'.format(len(image_paths)))
""")

add_code("""# Custom Dataset Class
class WeatherDataset(Dataset):
    def __init__(self, im_paths, transform=None):
        self.image_paths = im_paths
        self.transform = transform
        
        labels_str = []
        for p in self.image_paths:
            basename = os.path.basename(p)
            class_name = ''.join([c for c in basename.split('.')[0] if not c.isdigit()])
            labels_str.append(class_name)
            
        self.classes = sorted(list(set(labels_str)))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.labels = [self.class_to_idx[cls_name] for cls_name in labels_str]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        label = self.labels[idx]
        return image, label
""")

add_code("""# Normalization values
dt_mean = (0.4919, 0.4798, 0.4642)
dt_std = (0.2655, 0.2441, 0.2893)
normalizer = transforms.Normalize(mean=dt_mean, std=dt_std)

# Target shape is 240x240
train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=10),
    transforms.Resize((260, 260)),
    transforms.RandomCrop((240, 240)),
    transforms.RandomHorizontalFlip(p=0.2),
    transforms.ToTensor(),
    normalizer
])

valid_transform = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.CenterCrop((240, 240)),
    transforms.ToTensor(),
    normalizer
])

test_transform = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.CenterCrop((240, 240)),
    transforms.ToTensor(),
    normalizer
])
""")

add_code("""# Dataset splits
# We define the dataset with the test transformations (in principle)
full_dataset = WeatherDataset(image_paths, transform=test_transform)

n_valid = round(0.2 * len(full_dataset))
n_test = n_valid
n_train = len(full_dataset) - n_valid - n_test
train_dataset, valid_dataset, test_dataset = random_split(full_dataset,
                                                 [n_train, n_valid, n_test])

train_dataset = copy.deepcopy(train_dataset)
# The Subset class doesn't have transform, but dataset does. We need to apply it correctly.
# Subset wrap it, so train_dataset.dataset is WeatherDataset
train_dataset.dataset.transform = train_transform
valid_dataset.dataset.transform = valid_transform
test_dataset.dataset.transform = test_transform
""")

add_code("""# Dataloaders
num_workers = 2
batch_size = 64

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

loaders = {'train': train_loader, 'valid': valid_loader, 'test': test_loader}
""")

add_md("### Custom CNN")

add_code("""class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 4 Convolutional blocks
        # 1. Input: 3x240x240 -> Output: 32x120x120
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        
        # 2. Input: 32x120x120 -> Output: 64x60x60
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # 3. Input: 64x60x60 -> Output: 128x30x30
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # 4. Input: 128x30x30 -> Output: 256x15x15
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        
        self.maxpool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(p=0.4)
        
        self.fc1 = nn.Linear(256 * 15 * 15, 512)
        self.fc2 = nn.Linear(512, 4)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = self.maxpool(F.relu(self.conv3(x)))
        x = self.maxpool(F.relu(self.conv4(x)))
        x = x.reshape(-1, 256 * 15 * 15)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.logsoftmax(self.fc2(x))
        return x

model = CNN().to(DEVICE)
print(model)
""")

add_code("""criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.03, weight_decay=1e-4) # Added minor weight decay for regularization
FILENAME = 'L2_CNN_model.pt'

n_epochs = 40
model, tr_data = hp.train(n_epochs, loaders, model, optimizer, criterion, FILENAME)
""")

add_code("""# See the training plot
model, optimizer, checkpoint = hp.trained_load(FILENAME, model, optimizer)
hp.plot_checkpoint(checkpoint)
""")

add_code("""# Test the custom CNN model
accuracy, test_loss, outputs = hp.do_test(model, loaders, criterion)
print('Accuracy: {:.2f}'.format(accuracy))
""")

add_md("### Transfer learning with ResNet18")

add_code("""# Free up memory
del model, optimizer
torch.cuda.empty_cache()

model_transfer = models.resnet18(weights=ResNet18_Weights.DEFAULT)

# Freezing gradients
for param in model_transfer.parameters():
    param.requires_grad = False

# Replace classifier (unfrozen)
n_inputs_cls = model_transfer.fc.in_features
model_transfer.fc = nn.Linear(n_inputs_cls, 4)
model_transfer = model_transfer.to(DEVICE)
print(model_transfer)
""")

add_code("""FILENAME_TF = 'L2_transfer_model.pt'
optimizer_tf = torch.optim.SGD(model_transfer.fc.parameters(), lr=0.01)
criterion_tf = nn.CrossEntropyLoss()

n_epochs_tf = 15
model_transfer, tr_data_tf = hp.train(n_epochs_tf, loaders, model_transfer, optimizer_tf, criterion_tf, FILENAME_TF)
hp.plot_checkpoint({'tr_loss_list': tr_data_tf[0], 'vl_loss_list': tr_data_tf[1]})

accuracy_tf, test_loss_tf, outputs_tf = hp.do_test(model_transfer, loaders, criterion_tf)
print('Transfer Learning Accuracy: {:.2f}'.format(accuracy_tf))
""")

add_md("### Additional Exercises")
add_md("""
**1. Loss functions**:
`NLLLoss` paired with `LogSoftmax` is exactly equivalent to `CrossEntropyLoss` when calculating the final loss computationally. `CrossEntropyLoss` implicitly groups the two operations: calculating the logarithm of the softmax, and calculating the negative log likelihood. We can use them interchangeably as long as we make sure not to apply a LogSoftmax when using CrossEntropyLoss. `CrossEntropyLoss` is considered more convenient and mathematically stable.

**2. Dropout analysis**:
With Dropout layer inserted in the dense classification part, randomly dropping standard activations forces the network not to rely overwhelmingly on specific neurons or paths from the high-dimensional feature maps (e.g. 256*15*15), which heavily combats overfitting. Thus, the training loss curves track closer to validation loss curves (i.e. mitigating the symptom where training loss plummets while validation loss stagnates and rises). In fact, if dropout is quite high, validation loss could be slightly lower than training loss because validation mode completely disables dropout (all nodes participate simultaneously), rendering the network slightly more robust in eval phase.

**3. Transfer learning trade-offs**:
Advantages of freezing deep layers: Huge savings in computation, since gradients are not propagated backwards through convolutions. We avoid catastrophic forgetting of learned shapes from ImageNet dataset. Fast to train.
Disadvantages: The filters might be optimized for very different tasks and classes. Specifically, domain specific tasks (like specific medical imagery or microscopic cells) might suffer when layers are frozen.
Fine-tuning scenarios: If we had an enormous weather image dataset, unfreezing the last few convolutional layers to adjust their weights to weather-specific patterns is highly advised to get higher acc.

**4. Learning rate impact**:
We will now resume the training on the custom CNN checkpoint with 0.1x LR! Look at the code cell below.
""")

add_code("""# Part 4 - Training custom CNN resumed with 0.1x LR
model_custom = CNN().to(DEVICE)
optimizer_custom = torch.optim.SGD(model_custom.parameters(), lr=0.003)
model_custom, optimizer_custom, checkpoint_custom = hp.trained_load(FILENAME, model_custom, optimizer_custom)

print("Resuming training with 10x smaller learning rate (lr=0.003)...")
model_custom, tr_data_small_lr = hp.train(10, loaders, model_custom, optimizer_custom, criterion, FILENAME)
hp.plot_checkpoint({'tr_loss_list': tr_data_small_lr[0], 'vl_loss_list': tr_data_small_lr[1]})
""")

add_md("""**Observations on Smaller Learning Rate:**
By resuming training with a 10x smaller learning rate, we usually see that the loss, which had potentially plateaued during the initial training phase, might experience a small drop. The smaller learning rate allows the model's weights to settle into a finer minimum in the loss landscape, rather than bouncing around it. Overfitting might slightly increase if validation loss does not drop simultaneously.
""")

nb['cells'] = new_cells
with open('LESSON_2B.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
