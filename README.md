# Face Mask Detection - VS Code Project (Ready to run)

This project detects whether a person is wearing a face mask using a CNN and OpenCV for real-time detection.

## Contents
- `dataset/` — put your dataset here (see structure below)
- `model/` — saved model will be placed here after training
- `train_mask_detector.py` — training script
- `detect_mask_video.py` — real-time webcam detection script
- `prepare_dataset.py` — optional helper to split raw folders into train/test

## Dataset expected (simple):
Place images in this structure (recommended for quick start):
```
project_root/
  dataset/
    with_mask/
      img1.jpg
      img2.jpg
      ...
    without_mask/
      img1.jpg
      ...
```
Or use a `raw_dataset/with_mask` and `raw_dataset/without_mask` then run `prepare_dataset.py` to create `dataset/train` and `dataset/test` splits.

## Quick start (Windows)
1. Open **Anaconda Prompt** as Administrator.
2. Create and activate environment (recommended):
   ```bash
   conda create -n maskenv python=3.10 -y
   conda activate maskenv
   ```
3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
4. Put your dataset into `dataset/with_mask` and `dataset/without_mask` (as above).
5. Train the model:
   ```bash
   python train_mask_detector.py
   ```
   Model will be saved to `model/mask_detector_model.h5`.
6. Run real-time detection (after training):
   ```bash
   python detect_mask_video.py
   ```
   Press `q` in the video window to quit.

## Notes
- If you have issues installing TensorFlow, try using CPU-only version or use Google Colab.
- For mobile/Android deployment convert saved model to `.tflite` (optional).
