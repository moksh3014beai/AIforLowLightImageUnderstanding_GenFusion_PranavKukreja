import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler # For speed!
from tqdm import tqdm
import numpy as np

SAVE_DIR = './weights'
os.makedirs(SAVE_DIR, exist_ok=True)



# --- CONFIG ---
INPUTS_DIR = './dataset/inputs'
TARGETS_DIR = './dataset/targets'
SAVE_DIR = './weights'
BATCH_SIZE = 16  # Adjust based on GPU VRAM
LR = 1e-4
NUM_EPOCHS = 50
IMG_SIZE = (256, 256)

# --- 1. DATASET ---
class UnderwaterDataset(Dataset):
    def __init__(self, inputs_dir, targets_dir, transform=None):
        self.inputs_dir = inputs_dir
        self.targets_dir = targets_dir
        self.transform = transform
        self.images = sorted(os.listdir(inputs_dir))
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        in_path = os.path.join(self.inputs_dir, img_name)
        tar_path = os.path.join(self.targets_dir, img_name)
        
        # Robust loading
        try:
            input_img = cv2.imread(in_path)
            target_img = cv2.imread(tar_path)
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading {img_name}: {e}")
            return torch.zeros(3, *IMG_SIZE), torch.zeros(3, *IMG_SIZE)

        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)
        return input_img, target_img

# --- 2. MODEL (UNet with Residual Blocks) ---
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.LeakyReLU(0.2),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.LeakyReLU(0.2)
        )
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = ConvBlock(3, 32)
        self.e2 = ConvBlock(32, 64)
        self.e3 = ConvBlock(64, 128)
        self.pool = nn.MaxPool2d(2)
        
        self.bottleneck = ConvBlock(128, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.d3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.d1 = ConvBlock(64, 32)
        
        self.final = nn.Conv2d(32, 3, 1)

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(self.pool(x1))
        x3 = self.e3(self.pool(x2))
        
        b = self.bottleneck(self.pool(x3))
        
        x_up = self.up3(b)
        x_up = torch.cat([x_up, x3], dim=1)
        x_up = self.d3(x_up)
        
        x_up = self.up2(x_up)
        x_up = torch.cat([x_up, x2], dim=1)
        x_up = self.d2(x_up)
        
        x_up = self.up1(x_up)
        x_up = torch.cat([x_up, x1], dim=1)
        x_up = self.d1(x_up)
        
        return torch.sigmoid(self.final(x_up))

# --- 3. TRAINING ---
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting training on {device}...")
    
    transform = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize(IMG_SIZE), transforms.ToTensor()
    ])
    
    ds = UnderwaterDataset(INPUTS_DIR, TARGETS_DIR, transform)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.L1Loss() # Charbonnier Loss proxy
    scaler = torch.amp.GradScaler('cuda')   # For Mixed Precision

    for epoch in range(NUM_EPOCHS):
        model.train()
        loop = tqdm(dl, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        epoch_loss = 0
        
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            
            with autocast():
                pred = model(x)
                loss = criterion(pred, y)
                
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        # Save checkpoints
        torch.save(model.state_dict(), f"{SAVE_DIR}/model_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    main()