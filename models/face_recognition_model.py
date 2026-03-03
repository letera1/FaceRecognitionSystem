"""Face recognition model"""
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from config import Config
from utils.face_detector import FaceDetector
from utils.data_manager import DataManager

class FaceRecognitionModel:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.face_detector = FaceDetector()
        
    def prepare_training_data(self, faces, labels):
        """Prepare data for training"""
        processed_faces = []
        
        for face in faces:
            # Preprocess face
            processed = self.face_detector.preprocess_face(face)
            # Flatten
            processed_faces.append(processed.ravel())
        
        X = np.array(processed_faces)
        y = self.label_encoder.fit_transform(labels)
        
        return X, y
    
    def train_knn(self, X, y):
        """Train KNN model"""
        self.model = KNeighborsClassifier(
            n_neighbors=Config.KNN_NEIGHBORS,
            weights='distance',
            metric='euclidean'
        )
        self.model.fit(X, y)
        
    def train_svm(self, X, y):
        """Train SVM model"""
        self.model = SVC(
            kernel='rbf',
            probability=True,
            gamma='scale'
        )
        self.model.fit(X, y)
    
    def train(self, model_type='knn'):
        """Train the face recognition model"""
        # Load all faces
        faces, labels = DataManager.load_all_faces()
        
        if len(faces) == 0:
            raise ValueError("No training data available")
        
        # Prepare data
        X, y = self.prepare_training_data(faces, labels)
        
        # Train model
        if model_type == 'knn':
            self.train_knn(X, y)
        elif model_type == 'svm':
            self.train_svm(X, y)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Save model
        self.save_model()
        
        return len(faces), len(set(labels))
    
    def predict(self, face_image):
        """Predict identity from face image"""
        if self.model is None:
            self.load_model()
        
        # Preprocess
        processed = self.face_detector.preprocess_face(face_image)
        X = processed.ravel().reshape(1, -1)
        
        # Predict
        prediction = self.model.predict(X)[0]
        
        # Get confidence if available
        confidence = 0.0
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X)[0]
            confidence = np.max(proba)
        
        # Decode label
        label = self.label_encoder.inverse_transform([prediction])[0]
        
        return label, confidence
    
    def save_model(self):
        """Save trained model"""
        joblib.dump(self.model, Config.MODEL_PATH)
        joblib.dump(self.label_encoder, Config.LABEL_ENCODER_PATH)
    
    def load_model(self):
        """Load trained model"""
        if not os.path.exists(Config.MODEL_PATH):
            raise FileNotFoundError("Model not found. Please train the model first.")
        
        self.model = joblib.load(Config.MODEL_PATH)
        self.label_encoder = joblib.load(Config.LABEL_ENCODER_PATH)

import os
