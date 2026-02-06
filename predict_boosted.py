import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchvision import transforms

# --- CONFIGURATION ---
INPUT_FOLDER = "./dataset/inputs"
OUTPUT_FOLDER = "./submission_output_boosted" # New folder for boosted results
MODEL_PATH = "./weights/model_epoch_50.pth"   # Your best model
IMG_SIZE = (256, 256)

# --- MODEL DEFINITION (Must match exactly) ---
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

# --- SHARPENING KERNEL (Boosts SSIM) ---
def apply_sharpening(img):
    kernel = np.array([[0, -0.5, 0],
                       [-0.5, 3, -0.5],
                       [0, -0.5, 0]])
    return cv2.filter2D(img, -1, kernel)

# --- INFERENCE ---
def predict_boosted():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    print(f"Loading {MODEL_PATH}...")
    model = UNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor()
    ])

    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(('.jpg', '.png'))]
    print(f"Boosting {len(files)} images with TTA...")

    with torch.no_grad():
        for file_name in tqdm(files):
            img_path = os.path.join(INPUT_FOLDER, file_name)
            original_img = cv2.imread(img_path)
            if original_img is None: continue
            
            orig_h, orig_w = original_img.shape[:2]
            img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            
            # 1. Prepare Inputs (Normal & Flipped)
            t_img = transform(img_rgb).unsqueeze(0).to(device)
            t_img_flip = torch.flip(t_img, [3]) # Flip horizontally
            
            # 2. Predict Both
            out_normal = model(t_img)
            out_flip = model(t_img_flip)
            
            # 3. Un-flip the second one and Average
            out_flip_back = torch.flip(out_flip, [3])
            final_tensor = (out_normal + out_flip_back) / 2.0
            
            # 4. Post-Process
            output = final_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            output = np.clip(output * 255, 0, 255).astype(np.uint8)
            output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            
            # 5. Resize & Sharpen
            resized = cv2.resize(output_bgr, (orig_w, orig_h))
            
            # Mix 90% AI + 10% Sharpening (Safe boost for SSIM)
            sharpened = apply_sharpening(resized)
            final_output = cv2.addWeighted(resized, 0.9, sharpened, 0.1, 0)
            
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, file_name), final_output)

    print(f"Done! Boosted images in {OUTPUT_FOLDER}")

if __name__ == "__main__":
    predict_boosted()