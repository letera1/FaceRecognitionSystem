# Modern Face Recognition System

Production-grade face recognition system with deep learning embeddings.

## Features

- Deep learning face embeddings (FaceNet/dlib)
- Real-time face detection and recognition
- No retraining needed for new users
- Modern web interface
- Comprehensive Jupyter notebooks for training

## Quick Start

```bash
# Install dependencies
pip install -r requirements_production.txt

# Run Jupyter notebook for training
jupyter notebook notebooks/production_face_recognition.ipynb

# Run web application
python src/app.py
```

## Project Structure

```
face-recognition-system/
├── src/                          # Source code
│   ├── app.py                   # Flask application
│   ├── config.py                # Configuration
│   ├── models/                  # ML models
│   │   ├── face_recognition_model.py
│   │   └── deep_face_model.py
│   ├── utils/                   # Utilities
│   │   ├── face_detector.py
│   │   └── data_manager.py
│   └── scripts/                 # Helper scripts
│       └── generate_sample_data.py
├── notebooks/                    # Jupyter notebooks
│   ├── production_face_recognition.ipynb
│   └── face_recognition_complete.ipynb
├── templates/                    # HTML templates
│   └── home.html
├── data/                        # Data storage
│   ├── faces/                   # User face images
│   ├── attendance/              # Attendance records
│   └── database/                # Vector database
├── models/                      # Saved models
├── logs/                        # System logs
├── requirements_production.txt  # Dependencies
├── PRODUCTION_GUIDE.md         # Technical guide
└── README.md                   # This file
```

## System Architecture

- **Face Detection**: MTCNN
- **Embeddings**: FaceNet (512-D)
- **Similarity**: Cosine similarity
- **Storage**: FAISS vector database
- **Framework**: PyTorch + Flask

## Documentation

- [Production Guide](PRODUCTION_GUIDE.md) - Complete technical documentation
- [Jupyter Notebook](notebooks/production_face_recognition.ipynb) - Training workflow

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

## License

MIT
