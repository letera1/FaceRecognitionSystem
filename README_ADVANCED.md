# Advanced Face Recognition Attendance System

A modern, production-ready face recognition system with improved architecture, better UI, and comprehensive training tools.

## Features

- **Modular Architecture**: Clean separation of concerns with utils, models, and config
- **Advanced UI**: Modern gradient design with Bootstrap 5
- **Better Face Detection**: Improved preprocessing with histogram equalization
- **Multiple Models**: Support for KNN and SVM classifiers
- **Confidence Scoring**: Shows prediction confidence for better reliability
- **Training Notebook**: Jupyter notebook for model training and evaluation
- **Sample Data Generator**: Easy script to create training data
- **Comprehensive Logging**: Track attendance with timestamps
- **Scalable Structure**: Easy to extend and maintain

## Project Structure

```
FaceRecognitionSystem/
├── app_advanced.py              # Main Flask application
├── config.py                    # Configuration settings
├── requirements.txt             # Python dependencies
├── models/
│   ├── __init__.py
│   └── face_recognition_model.py  # ML model implementation
├── utils/
│   ├── __init__.py
│   ├── face_detector.py         # Face detection utilities
│   └── data_manager.py          # Data handling utilities
├── templates/
│   └── home_advanced.html       # Modern UI template
├── notebooks/
│   └── train_model.ipynb        # Training notebook
├── scripts/
│   └── generate_sample_data.py  # Sample data generator
├── data/
│   ├── faces/                   # User face images
│   ├── attendance/              # Attendance CSV files
│   ├── train/                   # Training data
│   └── test/                    # Test data
├── models/                      # Saved ML models
└── logs/                        # System logs
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the Haar Cascade file:
```
haarcascade_frontalface_default.xml
```

## Usage

### 1. Generate Sample Training Data

```bash
python scripts/generate_sample_data.py
```

Follow the prompts to add users and capture their face images.

### 2. Train Model (Using Notebook)

```bash
jupyter notebook notebooks/train_model.ipynb
```

Run all cells to:
- Load and visualize data
- Train KNN/SVM models
- Evaluate performance
- Generate confusion matrix
- Save trained model

### 3. Run the Application

```bash
python app_advanced.py
```

Access at: `http://127.0.0.1:5001`

### 4. Using the System

**Add New User:**
1. Enter name and ID in the form
2. System captures 50 face images automatically
3. Model retrains with new data

**Take Attendance:**
1. Click "Take Attendance" button
2. System recognizes faces in real-time
3. Shows confidence score
4. Automatically logs attendance
5. Press ESC to stop

## Configuration

Edit `config.py` to customize:

- `FACE_SIZE`: Face image dimensions (default: 128x128)
- `NUM_IMAGES_PER_USER`: Images per user (default: 50)
- `KNN_NEIGHBORS`: KNN parameter (default: 5)
- `CONFIDENCE_THRESHOLD`: Recognition threshold (default: 0.6)
- `PORT`: Server port (default: 5001)

## Key Improvements

1. **Better Architecture**: Modular design with separate concerns
2. **Enhanced Preprocessing**: Histogram equalization for better accuracy
3. **Confidence Scores**: Know how certain the prediction is
4. **Modern UI**: Gradient design with better UX
5. **Training Tools**: Jupyter notebook for experimentation
6. **Scalability**: Easy to add new features and models
7. **Error Handling**: Robust error management
8. **Documentation**: Comprehensive code comments

## API Endpoints

- `GET /` - Home dashboard
- `POST /add_user` - Register new user
- `GET /take_attendance` - Start attendance capture
- `GET /users` - Get all registered users (JSON)
- `GET /attendance_history` - Get attendance records (JSON)

## Model Performance

The system uses:
- **KNN Classifier**: Fast, simple, good for small datasets
- **SVM Classifier**: Better accuracy for larger datasets
- **Preprocessing**: Grayscale + Histogram Equalization + Normalization

Expected accuracy: 90-95% with good quality images

## Troubleshooting

**Camera not working:**
- Check camera permissions
- Try different camera index in `cv2.VideoCapture(0)`

**Low accuracy:**
- Capture more images per user (increase NUM_IMAGES_PER_USER)
- Ensure good lighting during capture
- Adjust CONFIDENCE_THRESHOLD

**Port already in use:**
- Change PORT in config.py
- Or kill process using the port

## Future Enhancements

- [ ] Deep learning models (CNN, FaceNet)
- [ ] Real-time video streaming
- [ ] Multi-camera support
- [ ] Export reports to PDF
- [ ] Email notifications
- [ ] REST API for mobile apps
- [ ] Face mask detection
- [ ] Emotion recognition

## License

MIT License
