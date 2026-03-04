"""Configuration settings for Face Recognition System"""
import os
from datetime import date

class Config:
    # Base directories
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')
    
    # Face data
    FACES_DIR = os.path.join(DATA_DIR, 'faces')
    TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'train')
    TEST_DATA_DIR = os.path.join(DATA_DIR, 'test')
    
    # Attendance
    ATTENDANCE_DIR = os.path.join(DATA_DIR, 'attendance')
    
    # Models
    MODEL_PATH = os.path.join(MODELS_DIR, 'face_recognition_model.pkl')
    DEEP_MODEL_PATH = os.path.join(MODELS_DIR, 'deep_face_model.pkl')
    FACE_ENCODER_PATH = os.path.join(MODELS_DIR, 'face_encoder.pkl')
    LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, 'label_encoder.pkl')
    
    # Face detection
    HAAR_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
    
    # Image settings
    FACE_SIZE = (128, 128)
    NUM_IMAGES_PER_USER = 50
    IMAGE_CAPTURE_INTERVAL = 5
    
    # Model settings
    KNN_NEIGHBORS = 5
    CONFIDENCE_THRESHOLD = 0.6
    FACE_DISTANCE_THRESHOLD = 0.6  # For face_recognition library
    USE_DEEP_LEARNING = True  # Use modern deep learning model
    
    # Flask settings
    SECRET_KEY = 'your-secret-key-here'
    DEBUG = True
    HOST = '127.0.0.1'
    PORT = 5001
    
    # Date formats
    DATE_FORMAT = date.today().strftime("%m_%d_%y")
    DATE_DISPLAY_FORMAT = date.today().strftime("%d-%B-%Y")
    
    @staticmethod
    def create_directories():
        """Create all necessary directories"""
        dirs = [
            Config.DATA_DIR,
            Config.MODELS_DIR,
            Config.LOGS_DIR,
            Config.FACES_DIR,
            Config.TRAIN_DATA_DIR,
            Config.TEST_DATA_DIR,
            Config.ATTENDANCE_DIR
        ]
        for directory in dirs:
            os.makedirs(directory, exist_ok=True)
