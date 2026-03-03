# Quick Start Guide

## Installation (5 minutes)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Verify Installation
```bash
python -c "import cv2, flask, sklearn; print('All packages installed!')"
```

## Usage

### Option A: Use Existing System (Simple)

1. **Start the advanced application:**
```bash
python app_advanced.py
```

2. **Open browser:**
```
http://127.0.0.1:5001
```

3. **Add users through the web interface**

### Option B: Generate Sample Data First (Recommended)

1. **Generate training data:**
```bash
python scripts/generate_sample_data.py
```

2. **Train model using Jupyter:**
```bash
jupyter notebook notebooks/train_model.ipynb
```

3. **Run the application:**
```bash
python app_advanced.py
```

## First Time Setup

### Add Your First User

1. Open `http://127.0.0.1:5001`
2. Fill in the form:
   - Name: `John`
   - ID: `101`
3. Click "Add User"
4. Look at the camera and let it capture 50 images
5. Wait for model training (10-30 seconds)
6. Done!

### Take Attendance

1. Click "Take Attendance" button
2. Face the camera
3. System recognizes you and logs attendance
4. Press ESC to stop
5. Check the dashboard for your record

## Troubleshooting

### Camera Not Working
```python
# Try different camera index
cap = cv2.VideoCapture(1)  # or 2, 3, etc.
```

### Port Already in Use
Edit `config.py`:
```python
PORT = 5002  # Change to any available port
```

### Low Recognition Accuracy
1. Ensure good lighting
2. Capture more images (increase NUM_IMAGES_PER_USER in config.py)
3. Lower CONFIDENCE_THRESHOLD in config.py

### Model Not Found Error
```bash
# Generate data and train model first
python scripts/generate_sample_data.py
```

## Key Features

✅ Modern gradient UI  
✅ Real-time face recognition  
✅ Confidence scoring  
✅ Automatic attendance logging  
✅ No duplicate entries per day  
✅ CSV export  
✅ Jupyter notebook for training  
✅ Sample data generator  

## Next Steps

1. **Customize UI**: Edit `templates/home_advanced.html`
2. **Adjust Settings**: Modify `config.py`
3. **Train Models**: Use `notebooks/train_model.ipynb`
4. **Add Features**: Extend `app_advanced.py`

## Comparison: Old vs New

| Feature | Old System | New System |
|---------|-----------|------------|
| Architecture | Monolithic | Modular |
| UI | Basic | Modern Gradient |
| Face Size | 50x50 | 128x128 |
| Preprocessing | Basic | Histogram Eq. |
| Confidence | No | Yes |
| Training Tool | No | Jupyter Notebook |
| Sample Data | Manual | Script |
| Code Quality | Mixed | Clean & Documented |
| Scalability | Limited | Better |

## Commands Cheat Sheet

```bash
# Start application
python app_advanced.py

# Generate sample data
python scripts/generate_sample_data.py

# Open Jupyter notebook
jupyter notebook notebooks/train_model.ipynb

# Check Python version
python --version

# List installed packages
pip list

# Create directories
python -c "from config import Config; Config.create_directories()"
```

## Support

For issues or questions:
1. Check `README_ADVANCED.md`
2. Review `SYSTEM_ARCHITECTURE.md`
3. Examine code comments
4. Test with sample data first
