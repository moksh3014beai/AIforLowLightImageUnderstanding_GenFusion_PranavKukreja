import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchvision import transforms

# --- CONFIGURATION ---
INPUT_FOLDER = "./dataset/inputs"       # Folder with the dark images to enhance
OUTPUT_FOLDER = "./submission_output"   # Where to save the results
MODEL_PATH = "./weights/model_epoch_50.pth" # <--- CHECK THIS FILENAME!
IMG_SIZE = (256, 256)                   # Must match training size

# --- MODEL DEFINITION (Must be EXACTLY the same as training) ---
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

# --- INFERENCE LOGIC ---
def predict():
    # 1. Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running inference on: {device}")
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # 2. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        print("Please check the filename in 'weights' folder.")
        return

    print(f"Loading weights from {MODEL_PATH}...")
    model = UNet().to(device)
    
    # Load weights
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error loading state dict: {e}")
        return

    model.eval()
    
    # 3. Get Files
    valid_exts = ('.jpg', '.png', '.jpeg', '.bmp')
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(valid_exts)]
    
    if len(files) == 0:
        print(f"ERROR: No images found in {INPUT_FOLDER}")
        return

    print(f"Processing {len(files)} images...")
    
    # 4. Processing Loop
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor()
    ])

    with torch.no_grad():
        for file_name in tqdm(files):
            img_path = os.path.join(INPUT_FOLDER, file_name)
            
            # Read Image
            original_img = cv2.imread(img_path)
            if original_img is None:
                continue
            
            # Keep original dimensions for final resize
            orig_h, orig_w = original_img.shape[:2]
            
            # Preprocess (BGR -> RGB -> Transform)
            img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            input_tensor = transform(img_rgb).unsqueeze(0).to(device)
            
            # Inference
            output_tensor = model(input_tensor)
            
            # Postprocess
            # (1, 3, 256, 256) -> (3, 256, 256) -> (256, 256, 3)
            output = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            # Scale back to 0-255
            output = np.clip(output * 255, 0, 255).astype(np.uint8)
            
            # Convert RGB back to BGR for OpenCV saving
            output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            
            # Resize BACK to original dimensions (Requirement for submission)
            final_output = cv2.resize(output_bgr, (orig_w, orig_h))
            
            # Save
            save_path = os.path.join(OUTPUT_FOLDER, file_name)
            cv2.imwrite(save_path, final_output)

    print(f"\nSUCCESS! Enhanced images saved to: {OUTPUT_FOLDER}")
    

if __name__ == "__main__":
    predict()