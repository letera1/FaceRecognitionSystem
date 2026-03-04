# Modern Project Structure

```
face-recognition-system/
│
├── src/                              # Source code
│   ├── __init__.py
│   ├── app.py                       # Flask application
│   ├── config.py                    # Configuration
│   │
│   ├── models/                      # ML models
│   │   ├── __init__.py
│   │   ├── face_recognition_model.py
│   │   └── deep_face_model.py
│   │
│   ├── utils/                       # Utilities
│   │   ├── __init__.py
│   │   ├── face_detector.py
│   │   └── data_manager.py
│   │
│   └── scripts/                     # Helper scripts
│       └── generate_sample_data.py
│
├── notebooks/                        # Jupyter notebooks
│   ├── production_face_recognition.ipynb
│   └── face_recognition_complete.ipynb
│
├── templates/                        # HTML templates
│   └── home.html
│
├── data/                            # Data storage
│   ├── faces/                       # User face images
│   │   ├── person_001/
│   │   ├── person_002/
│   │   └── ...
│   ├── attendance/                  # Attendance records
│   │   └── *.csv
│   ├── database/                    # Vector database
│   │   ├── face_vectors.index
│   │   ├── metadata.db
│   │   └── embeddings.npy
│   └── test/                        # Test data
│
├── models/                          # Saved models
│   ├── face_recognition_model.pkl
│   ├── deep_face_model.pkl
│   └── label_encoder.pkl
│
├── logs/                            # System logs
│   └── *.log
│
├── tests/                           # Unit tests
│   ├── test_face_detector.py
│   ├── test_embedder.py
│   └── test_app.py
│
├── docs/                            # Documentation
│   ├── API.md
│   ├── DEPLOYMENT.md
│   └── ARCHITECTURE.md
│
├── .env.example                     # Environment template
├── .gitignore                       # Git ignore
├── run.py                           # Application entry
├── setup.py                         # Package setup
├── requirements_production.txt      # Production deps
├── requirements.txt                 # Development deps
├── README.md                        # Main documentation
├── PRODUCTION_GUIDE.md             # Technical guide
└── STRUCTURE.md                    # This file
```

## Key Directories

### `/src` - Source Code
All application code organized by function:
- `app.py` - Flask web application
- `config.py` - Centralized configuration
- `models/` - ML model implementations
- `utils/` - Helper functions
- `scripts/` - Standalone scripts

### `/notebooks` - Jupyter Notebooks
Training and experimentation:
- Production training pipeline
- Model evaluation
- Data analysis

### `/data` - Data Storage
All data organized by type:
- `faces/` - Training images (folder per person)
- `attendance/` - CSV attendance logs
- `database/` - Vector database and metadata
- `test/` - Test datasets

### `/models` - Saved Models
Trained model artifacts:
- `.pkl` files for scikit-learn models
- `.pth` files for PyTorch models
- Label encoders and metadata

### `/templates` - Web Templates
HTML templates for Flask application

### `/logs` - System Logs
Application and error logs

## Running the System

```bash
# Development
python run.py

# Production
gunicorn -w 4 -b 0.0.0.0:5001 src.app:app
```

## Import Structure

```python
# Correct imports
from src.config import Config
from src.models.deep_face_model import DeepFaceModel
from src.utils import DataManager

# Run from project root
python run.py
```
