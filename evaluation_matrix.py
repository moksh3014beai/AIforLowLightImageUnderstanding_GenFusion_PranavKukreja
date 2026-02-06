import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm

# --- CONFIGURATION ---
PREDICTED_DIR = "./submission_output"  # Your generated images
TARGET_DIR = "./dataset/targets"       # The actual ground truth (clean images)

def calculate_metrics():
    # Get list of generated files
    pred_files = sorted(os.listdir(PREDICTED_DIR))
    
    psnr_scores = []
    ssim_scores = []
    
    print(f"Evaluating {len(pred_files)} images...")
    
    for file_name in tqdm(pred_files):
        # Construct paths
        pred_path = os.path.join(PREDICTED_DIR, file_name)
        target_path = os.path.join(TARGET_DIR, file_name)
        
        # Check if target exists (sometimes names don't match exactly)
        if not os.path.exists(target_path):
            # Try matching by index if filenames differ (e.g., input_01 vs target_01)
            # This is a fallback; usually filenames match.
            continue
            
        # Load images
        img_pred = cv2.imread(pred_path)
        img_target = cv2.imread(target_path)
        
        if img_pred is None or img_target is None:
            print(f"Error reading {file_name}")
            continue
            
        # Ensure sizes match (Crucial!)
        if img_pred.shape != img_target.shape:
            img_pred = cv2.resize(img_pred, (img_target.shape[1], img_target.shape[0]))
        
        # Calculate PSNR
        # Data range is 255 because images are uint8
        score_psnr = psnr(img_target, img_pred, data_range=255)
        
        # Calculate SSIM
        # multidichannel=True is required for RGB images
        # channel_axis=2 tells it the 3rd dimension is color
        score_ssim = ssim(img_target, img_pred, data_range=255, channel_axis=2)
        
        psnr_scores.append(score_psnr)
        ssim_scores.append(score_ssim)

    # --- FINAL REPORT ---
    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)
    
    print("\n" + "="*30)
    print("   FINAL EVALUATION REPORT   ")
    print("="*30)
    print(f"Total Images Evaluated: {len(psnr_scores)}")
    print(f"Average PSNR: {avg_psnr:.4f} dB  (Target: >20 dB)")
    print(f"Average SSIM: {avg_ssim:.4f}     (Target: >0.70)")
    print("="*30)

    # Save to a file for submission
    with open("evaluation_metrics.txt", "w") as f:
        f.write(f"Average PSNR: {avg_psnr:.4f}\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")
    print("Saved results to evaluation_metrics.txt")

if __name__ == "__main__":
    calculate_metrics()