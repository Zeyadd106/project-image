
# 🔥 Forest Fire, Smoke & Non-Fire Detection

A deep learning project for classifying forest images into three categories — **Fire**, **Smoke**, and **Non-Fire** — using two models: a custom CNN built from scratch and a MobileNetV2 transfer learning model.

---

## 📁 Repository Structure

```
├── data/
│   └── forest_dataset/
│       └── FOREST_FIRE_SMOKE_AND_NON_FIRE_DATASET/
│           ├── train/
│           │   ├── fire/
│           │   ├── non fire/
│           │   └── Smoke/
│           └── test/
│               ├── fire/
│               ├── non fire/
│               └── Smoke/
├── notebooks/
│   └── FireDataKaggle.ipynb
├── models/
│   ├── cnn_forest_fire_model.h5
│   └── mobilenetv2_forest_fire_model.h5
├── results/
│   └── (plots, confusion matrices, training curves)
├── README.md
└── requirements.txt
```

---

## 📊 Dataset

- **Source:** [Forest Fire, Smoke and Non-Fire Image Dataset](https://www.kaggle.com/datasets/amerzishminha/forest-fire-smoke-and-non-fire-image-dataset) on Kaggle
- **License:** CC0-1.0
- **Classes:** `fire` · `non fire` · `Smoke`
- **Size:**
  - Train: 10,800 images per class (32,400 total)
  - Test: 3,500 images per class (10,500 total)
- **Class Balance:** Perfectly balanced — no class weighting required

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd <repo-folder>
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Kaggle API credentials

To download the dataset automatically from the notebook, set your Kaggle credentials:
```bash
export KAGGLE_USERNAME="your_kaggle_username"
export KAGGLE_KEY="your_kaggle_api_key"
```
> You can find your API key at: https://www.kaggle.com/settings → API → Create New Token

### 4. Run the notebook
```bash
jupyter notebook notebooks/FireDataKaggle.ipynb
```
> The notebook will download and extract the dataset automatically on first run.

---

## 🔄 Preprocessing Pipeline

Applied to both models (shared pipeline):

| Step | Details |
|---|---|
| Corrupt image check | Scans all files using `imghdr`; reports unreadable images |
| Class balance check | Computes ratio; applies `class_weight` if imbalanced |
| Resize | All images resized to **64×64** pixels |
| Normalize | Pixel values rescaled to **[0, 1]** |
| Train/Val/Test split | **70% train / 15% validation / 15% test** |
| Augmentation (train only) | Rotation ±15°, horizontal flip, zoom 10%, brightness [0.7–1.3], color jitter |

---

## 🧠 Models

### Model 1 — CNN from Scratch

Built manually layer by layer using Keras Sequential API:

- Conv2D → ReLU → MaxPooling (×3 blocks)
- BatchNormalization
- Flatten → Dense → Dropout
- Output: Dense(3, activation='softmax')

### Model 2 — MobileNetV2 Transfer Learning

Built manually (no drag-and-drop):

- **Phase 1:** Load MobileNetV2 (ImageNet weights, no top head) → freeze all base layers → add custom head: `GlobalAveragePooling2D → Dense(128, ReLU) → Dropout(0.3) → Dense(3, Softmax)`
- **Phase 2 (Fine-tuning):** Unfreeze last few layers of MobileNetV2 → retrain with lower learning rate

---

## 📈 Evaluation

Each model is evaluated with:

- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix (heatmap)
- Training vs. Validation Loss & Accuracy curves (per epoch)

A **comparison table** is included at the end of the notebook covering:
- Train/Val Accuracy
- Training Time
- Trainable Parameters
- Final Test Performance

---

## 💾 Saved Models

Both trained models are saved in HDF5 format:
```
models/cnn_forest_fire_model.h5
models/mobilenetv2_forest_fire_model.h5
```

To load and use a saved model:
```python
from tensorflow.keras.models import load_model
model = load_model('models/mobilenetv2_forest_fire_model.h5')
predictions = model.predict(your_image_array)
```

---

## 🏁 Results Summary

| Model | Notes |
|---|---|
| CNN from Scratch | Trained from random initialization; good baseline performance |
| MobileNetV2 TL | Pre-trained ImageNet weights; higher accuracy, faster convergence |

> Transfer Learning achieved the best overall performance on the test set.

---

## 👥 Team

Computer Vision & Image Processing — Final Project
