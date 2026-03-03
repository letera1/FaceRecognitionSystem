# System Architecture

## Overview
```
┌─────────────────────────────────────────────────────────────┐
│                    Flask Web Application                     │
│                      (app_advanced.py)                       │
└───────────────────┬─────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
┌───────────────┐       ┌──────────────────┐
│  Face         │       │  Data            │
│  Recognition  │       │  Manager         │
│  Model        │       │                  │
└───────┬───────┘       └────────┬─────────┘
        │                        │
        ▼                        ▼
┌───────────────┐       ┌──────────────────┐
│  Face         │       │  Database        │
│  Detector     │       │  (CSV Files)     │
└───────────────┘       └──────────────────┘
```

## Component Details

### 1. Flask Web Application (app_advanced.py)
- **Purpose**: Main entry point, handles HTTP requests
- **Routes**:
  - `/` - Dashboard
  - `/add_user` - Register new user
  - `/take_attendance` - Capture attendance
  - `/users` - API endpoint for user list
  - `/attendance_history` - API endpoint for records

### 2. Face Recognition Model (models/face_recognition_model.py)
- **Algorithms**: KNN, SVM
- **Features**:
  - Training with multiple algorithms
  - Confidence scoring
  - Model persistence (joblib)
- **Input**: 128x128 grayscale face images
- **Output**: User label + confidence score

### 3. Face Detector (utils/face_detector.py)
- **Method**: Haar Cascade Classifier
- **Functions**:
  - `detect_faces()` - Find faces in image
  - `extract_face()` - Crop and resize face
  - `preprocess_face()` - Normalize for model

### 4. Data Manager (utils/data_manager.py)
- **Functions**:
  - `save_user_face()` - Store face images
  - `load_all_faces()` - Load training data
  - `save_attendance()` - Log attendance
  - `get_today_attendance()` - Retrieve records

### 5. Configuration (config.py)
- Centralized settings
- Directory paths
- Model parameters
- Flask settings

## Data Flow

### Registration Flow
```
User Input (Name, ID)
    ↓
Webcam Capture (50 images)
    ↓
Face Detection
    ↓
Face Extraction & Resize (128x128)
    ↓
Save to data/faces/[name]_[id]/
    ↓
Retrain Model
    ↓
Save Model (models/*.pkl)
```

### Attendance Flow
```
Start Attendance
    ↓
Webcam Stream
    ↓
Face Detection
    ↓
Face Extraction
    ↓
Preprocessing (Grayscale + Equalization)
    ↓
Model Prediction
    ↓
Confidence Check (>= 0.6)
    ↓
Save to CSV (if not already marked)
    ↓
Display on Dashboard
```

## File Structure
```
FaceRecognitionSystem/
│
├── app_advanced.py          # Main application
├── config.py                # Configuration
├── requirements.txt         # Dependencies
│
├── models/                  # ML Models
│   ├── __init__.py
│   ├── face_recognition_model.py
│   └── *.pkl               # Saved models
│
├── utils/                   # Utilities
│   ├── __init__.py
│   ├── face_detector.py
│   └── data_manager.py
│
├── templates/               # HTML Templates
│   └── home_advanced.html
│
├── notebooks/               # Jupyter Notebooks
│   └── train_model.ipynb
│
├── scripts/                 # Helper Scripts
│   └── generate_sample_data.py
│
├── data/                    # Data Storage
│   ├── faces/              # User images
│   ├── attendance/         # CSV files
│   ├── train/              # Training data
│   └── test/               # Test data
│
└── logs/                    # System logs
```

## Technology Stack

### Backend
- **Flask**: Web framework
- **OpenCV**: Computer vision
- **scikit-learn**: Machine learning
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation

### Frontend
- **Bootstrap 5**: UI framework
- **Material Icons**: Icons
- **HTML5/CSS3**: Structure & styling

### ML Pipeline
- **Haar Cascade**: Face detection
- **KNN/SVM**: Classification
- **Histogram Equalization**: Preprocessing
- **Joblib**: Model serialization

## Performance Considerations

### Optimization Strategies
1. **Image Preprocessing**: Grayscale + equalization reduces computation
2. **Face Size**: 128x128 balances accuracy and speed
3. **KNN**: Fast inference for real-time recognition
4. **Caching**: Model loaded once, reused for predictions

### Scalability
- **Current**: Handles 10-50 users efficiently
- **Recommended**: Up to 100 users with KNN
- **For larger**: Switch to SVM or deep learning (FaceNet)

## Security Considerations

1. **Data Privacy**: Face images stored locally
2. **Access Control**: No authentication (add if needed)
3. **Input Validation**: Form validation on client & server
4. **File Permissions**: Restrict access to data directories

## Future Enhancements

1. **Deep Learning**: CNN, FaceNet for better accuracy
2. **Live Streaming**: WebRTC for browser-based capture
3. **Authentication**: User login system
4. **API**: RESTful API for mobile apps
5. **Cloud Storage**: AWS S3 for face images
6. **Analytics**: Dashboard with charts and reports
7. **Multi-camera**: Support multiple camera feeds
8. **Face Mask Detection**: COVID-19 compliance
