import os
import cv2
import torch
import numpy as np
import lpips
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm

# --- CONFIGURATION ---
PREDICTED_DIR = "./submission_output_boosted"  # Your best images
TARGET_DIR = "./dataset/targets"               # Ground Truth
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- UCIQE CALCULATION FUNCTION (The Mathy Part) ---
def get_uciqe(img_bgr):
    # Convert to LAB color space
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)
    
    # Cast to float for math
    l = l.astype(np.float32)
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    
    # 1. Chroma (Saturation)
    chroma = np.sqrt(a**2 + b**2)
    mu_c = np.mean(chroma)
    
    # 2. Contrast of Luminance
    # (Difference between bottom 1% and top 1% of pixels)
    l_flat = l.flatten()
    delta_l = np.percentile(l_flat, 99) - np.percentile(l_flat, 1)
    
    # 3. Saturation (Average Saturation)
    # UCIQE formula constants (standard values)
    c1 = 0.4680
    c2 = 0.2745
    c3 = 0.2576
    
    uciqe = c1 * (np.std(chroma) / mu_c) + c2 * (delta_l / 255.0) + c3 * (mu_c / 255.0)
    return uciqe

def evaluate_all():
    # Setup LPIPS (Downloads weights on first run)
    print("Initializing LPIPS metric...")
    loss_fn_alex = lpips.LPIPS(net='alex').to(DEVICE)
    
    files = sorted(os.listdir(PREDICTED_DIR))
    
    # Metric Lists
    scores_psnr = []
    scores_ssim = []
    scores_lpips = []
    scores_uciqe = []
    
    print(f"Evaluating {len(files)} images for ALL metrics...")
    
    for file_name in tqdm(files):
        path_pred = os.path.join(PREDICTED_DIR, file_name)
        path_target = os.path.join(TARGET_DIR, file_name)
        
        if not os.path.exists(path_target): continue
        
        # Load Images
        img_pred = cv2.imread(path_pred)
        img_target = cv2.imread(path_target)
        if img_pred is None or img_target is None: continue
        
        # Resize if needed
        if img_pred.shape != img_target.shape:
            img_pred = cv2.resize(img_pred, (img_target.shape[1], img_target.shape[0]))

        # --- PSNR & SSIM ---
        scores_psnr.append(psnr(img_target, img_pred, data_range=255))
        scores_ssim.append(ssim(img_target, img_pred, data_range=255, channel_axis=2))
        
        # --- UCIQE (Calculated on Prediction Only) ---
        scores_uciqe.append(get_uciqe(img_pred))
        
        # --- LPIPS (Requires Tensors) ---
        # Convert to Tensor, Normalize to [-1, 1], Move to GPU
        t_pred = lpips.im2tensor(img_pred).to(DEVICE)
        t_target = lpips.im2tensor(img_target).to(DEVICE)
        
        with torch.no_grad():
            d = loss_fn_alex(t_pred, t_target)
            scores_lpips.append(d.item())

    # --- FINAL REPORT ---
    print("\n" + "="*40)
    print("   FULL METRICS REPORT   ")
    print("="*40)
    print(f"PSNR  : {np.mean(scores_psnr):.4f}  (Goal: >26.2)")
    print(f"SSIM  : {np.mean(scores_ssim):.4f}  (Goal: >0.90)")
    print(f"LPIPS : {np.mean(scores_lpips):.4f} (Goal: <0.095)") # Lower is better
    print(f"UCIQE : {np.mean(scores_uciqe):.4f} (Goal: >0.42)")
    print("="*40)

if __name__ == "__main__":
    evaluate_all()