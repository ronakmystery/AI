import torch
import torch.nn as nn
import torch.nn.functional as F

# ✅ Define CNN Model for Mel Spectrograms
class ImprovedBirdSoundCNN(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedBirdSoundCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)  # (128x128 → 64x64)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # (64x64 → 32x32)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # (32x32 → 16x16)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)  # (16x16 → 8x8)

        self.fc1 = nn.Linear(256 * 8 * 8, 256)  # Adjusted for 128x128 input
        self.dropout = nn.Dropout(0.3)  # Reduce overfitting
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 128x128 → 64x64
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))  # 64x64 → 32x32
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))  # 32x32 → 16x16
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))  # 16x16 → 8x8
        
        x = x.view(x.shape[0], -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  
        return x