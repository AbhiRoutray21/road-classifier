# 🛣️ Road Image Classifier
### Binary Classification: Road vs Not Road
#### Built with MobileNetV2 Transfer Learning | CPU-Optimised

---

## 📁 Project Structure

```
road_classifier/
│
├── data/
│   ├── train/
│   │   ├── road/          ← Put road training images here
│   │   └── not_road/      ← Put non-road training images here
│   ├── val/
│   │   ├── road/          ← Put road validation images here
│   │   └── not_road/      ← Put non-road validation images here
│   └── test/              ← Optional: unlabelled test images
│
├── models/                ← Trained model saved here automatically
│
├── requirements.txt       ← Python dependencies
├── prepare_data.py        ← Helps organise your dataset
├── train.py               ← Train the model
├── predict.py             ← Run inference (CLI)
├── app.py                 ← Web UI (drag & drop images)
└── README.md
```

---

## ⚙️ Setup (Step by Step)

### Step 1: Install Python
Make sure you have **Python 3.9 or 3.10** installed.
Download: https://www.python.org/downloads/

### Step 2: Create Virtual Environment
Open Command Prompt or PowerShell inside the `road_classifier` folder:

```bash
# Create virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# You should see (venv) in your terminal now
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```
> ⏱️ This takes 5-10 minutes depending on internet speed

---

## 📸 Prepare Your Dataset

### Option A — You have images in folders
```bash
# If you have a folder of road images and a folder of other images:
python prepare_data.py --road C:\path\to\road_images --notroad C:\path\to\other_images

# This automatically splits them 80% train / 20% val
```

### Option B — Manually place images
Manually copy images into the correct folders:
- Road images → `data/train/road/` and `data/val/road/`
- Non-road images → `data/train/not_road/` and `data/val/not_road/`

Then verify:
```bash
python prepare_data.py --verify
```

### Where to get free datasets?
```bash
python prepare_data.py --sources
```

### Minimum recommended:
| Class    | Minimum | Recommended |
|----------|---------|-------------|
| road     | 200     | 500–1000    |
| not_road | 200     | 500–1000    |

> 💡 You can take photos yourself using your phone!

---

## 🚀 Training

```bash
python train.py
```

### Expected training time on your laptop (Intel Core Ultra 7 155U):
| Dataset Size | Estimated Time |
|---|---|
| 400 images  | ~15–20 minutes  |
| 1000 images | ~40–60 minutes  |
| 3000 images | ~2–3 hours      |

> 💡 Tip: Start with a small dataset (400 images) to verify everything works, then add more images for better accuracy.

Training will automatically:
- Download MobileNetV2 pretrained weights (first time only)
- Train for 15 epochs
- Save the best model to `models/road_classifier.pth`
- Show training curves (accuracy & loss graphs)

---

## 🔍 Running Predictions

### Single image:
```bash
python predict.py --image path/to/your/image.jpg
```

### Entire folder of images:
```bash
python predict.py --folder path/to/image/folder
```

### Live webcam:
```bash
python predict.py --webcam
# Press Q to quit
```

---

## 🌐 Web UI (Easiest Way)

```bash
python app.py
```
Opens a browser window where you can drag & drop any image!

---

## ⚡ Performance Tips for Your Laptop

1. **Keep batch_size = 16** (set in train.py CONFIG) — prevents RAM overuse
2. **Start with freeze_layers = True** (faster training)
3. **Close other apps** while training (browser, etc.)
4. **Plug in charger** — Windows throttles CPU on battery
5. **Set Power Mode to Best Performance** in Windows settings

---

## 🎯 Expected Accuracy

With a decent dataset:
- 500 images total → ~85–90% accuracy
- 1000 images total → ~90–94% accuracy
- 3000+ images total → ~95%+ accuracy

---

## 🛠️ Tweaking the Model (Advanced)

Edit the `CONFIG` dictionary in `train.py`:

```python
CONFIG = {
    "num_epochs"    : 15,       # increase for better accuracy
    "batch_size"    : 16,       # lower if RAM issues
    "learning_rate" : 0.001,
    "freeze_layers" : True,     # set False to fine-tune all layers (slower)
    "image_size"    : 224,
}
```

---

## ❓ Common Issues

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `No images found` | Check folder structure with `prepare_data.py --verify` |
| Training very slow | Normal on CPU. Reduce dataset size or epochs |
| Low accuracy | Add more diverse images, increase epochs |
| Out of memory | Reduce batch_size to 8 |

---

*Made with ❤️ | MobileNetV2 + PyTorch | Optimised for CPU*
