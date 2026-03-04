# Installation Guide

## Quick Install

```bash
# 1. Install basic dependencies
pip install -r requirements.txt

# 2. Install face_recognition (optional, for deep learning)
pip install face-recognition

# 3. Run the application
python run.py
```

## System Status

The system works in TWO modes:

### Mode 1: Basic Mode (Current)
- Uses traditional face detection (Haar Cascade)
- KNN/SVM classification
- Works without face_recognition library
- ✅ Currently working

### Mode 2: Deep Learning Mode (Requires installation)
- Uses dlib + face_recognition
- 128-D embeddings
- Higher accuracy
- Requires: `pip install face-recognition`

## Verify Installation

```bash
python -c "from src.app import app; print('✅ System ready')"
```

## Run Application

```bash
python run.py
```

Access at: http://127.0.0.1:5001

## Training

Use Jupyter notebooks:
```bash
jupyter notebook notebooks/face_recognition_complete.ipynb
```

## Troubleshooting

**face_recognition not found:**
```bash
pip install cmake
pip install dlib
pip install face-recognition
```

**Import errors:**
- Make sure you run from project root
- Python 3.8+ required
