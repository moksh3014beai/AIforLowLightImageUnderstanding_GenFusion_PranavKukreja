

# AI for Low-Light Image Understanding & Underwater Tracking
### Team: GenFusion
**Pranav Kukreja ,Moksh Rapal,Varun Chawla ,Vansh Chawla ** – Model Training & Optimization , Image Enhancement
**Event:** AI Summit Hackathon 2026 | **Track:** Computer Vision

---

## 📖 Overview
This repository contains our solution for the "AI for Low-Light Image Understanding" challenge. Our pipeline addresses two critical tasks in underwater and low-light environments:
1.  **Image Enhancement:** Restoring visibility in dark, low-contrast underwater images using a deep learning-based U-Net.


---

## 🚀 Methodology

### Phase 1: Low-Light Image Enhancement (Task 1)
We implemented a **U-Net** based architecture with skip connections to preserve high-frequency details while recovering color and illumination.
* **Architecture:** Custom U-Net with learnable upsampling.
* **Loss Function:** Charbonnier Loss (Robust L1) + SSIM (Structural Similarity) to ensure perceptual quality.
* **Optimization:** Mixed Precision Training (AMP) for faster convergence on RTX 4060 GPU.
* **Input/Output:** Processes 256x256 images, outputting enhanced RGB images.

First Full Evaluation results:
   FULL METRICS REPORT   
========================================
PSNR  : 22.1877  (Goal: >26.2)
SSIM  : 0.7978  (Goal: >0.90)
LPIPS : 0.3313 (Goal: <0.095)
UCIQE : 0.4223 (Goal: >0.42)

**weights_old is for older and more stable models and weights is for newer versions**



## 📂 Project Structure

```text
AIforLowLightImageUnderstanding_GenFusion/
│
├── dataset/                  # Dataset for Task 1 (Inputs/Targets)
├── dataset_tracking/         # Dataset for Task 2 (Video Sequences)
├── weights/                  # Saved Model Checkpoints (.pth, .pt)
├── submission_output/        # Final generated results
│
├── train_phase1.py           # Training script for Image Enhancement (U-Net)
├── predict_submission.py     # Inference script for Task 1 (Generates Final Images)
│
├── prepare_tracking_data.py  # Converts GroundTruth.txt to YOLO format
├── train_tracker.py          # Fine-tuning script for YOLOv8
├── track_phase2.py           # Inference script for Object Tracking
│
├── requirements.txt          # Python dependencies
└── README.md                 # Project Documentation
