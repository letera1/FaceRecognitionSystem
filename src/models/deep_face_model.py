"""Modern Deep Learning Face Recognition Model using face_recognition library"""
import os
import sys
import numpy as np
import cv2
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("Warning: face_recognition not available. Install with: pip install face-recognition")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

class DeepFaceModel:
    """Modern face recognition using face_recognition library (dlib-based)"""
    
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.face_encodings = []
        self.labels = []
        
    def extract_face_encoding(self, image):
        """Extract 128-dimensional face encoding using dlib"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect face locations
        face_locations = face_recognition.face_locations(rgb_image, model="hog")
        
        if len(face_locations) == 0:
            return None
        
        # Get face encodings (128-dimensional vector)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        if len(face_encodings) > 0:
            return face_encodings[0]
        return None
    
    def prepare_training_data(self, faces, labels):
        """Extract face encodings from all training images"""
        encodings = []
        valid_labels = []
        
        print("Extracting face encodings...")
        for i, face in enumerate(faces):
            encoding = self.extract_face_encoding(face)
            if encoding is not None:
                encodings.append(encoding)
                valid_labels.append(labels[i])
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(faces)} images")
        
        X = np.array(encodings)
        y = self.label_encoder.fit_transform(valid_labels)
        
        return X, y
    
    def train_svm(self, X, y):
        """Train SVM classifier on face encodings"""
        self.model = SVC(
            kernel='linear',
            probability=True,
            C=1.0,
            random_state=42
        )
        self.model.fit(X, y)
        print(f"SVM trained with {len(X)} samples")
    
    def train_knn(self, X, y):
        """Train KNN classifier on face encodings"""
        self.model = KNeighborsClassifier(
            n_neighbors=min(5, len(np.unique(y))),
            weights='distance',
            metric='euclidean'
        )
        self.model.fit(X, y)
        print(f"KNN trained with {len(X)} samples")
    
    def train(self, faces, labels, model_type='svm'):
        """Train the face recognition model"""
        if len(faces) == 0:
            raise ValueError("No training data available")
        
        # Prepare data
        X, y = self.prepare_training_data(faces, labels)
        
        if len(X) == 0:
            raise ValueError("No valid face encodings extracted")
        
        # Train model
        if model_type == 'svm':
            self.train_svm(X, y)
        elif model_type == 'knn':
            self.train_knn(X, y)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Save model
        self.save_model()
        
        return len(X), len(np.unique(y))
    
    def predict(self, face_image):
        """Predict identity from face image"""
        if self.model is None:
            self.load_model()
        
        # Extract encoding
        encoding = self.extract_face_encoding(face_image)
        
        if encoding is None:
            return None, 0.0
        
        # Predict
        X = encoding.reshape(1, -1)
        prediction = self.model.predict(X)[0]
        
        # Get confidence
        confidence = 0.0
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X)[0]
            confidence = np.max(proba)
        
        # Decode label
        label = self.label_encoder.inverse_transform([prediction])[0]
        
        return label, confidence
    
    def compare_faces(self, known_encoding, face_image, tolerance=0.6):
        """Compare face with known encoding"""
        unknown_encoding = self.extract_face_encoding(face_image)
        
        if unknown_encoding is None:
            return False, 1.0
        
        # Calculate face distance
        face_distance = face_recognition.face_distance([known_encoding], unknown_encoding)[0]
        
        # Check if match
        is_match = face_distance <= tolerance
        
        return is_match, face_distance
    
    def save_model(self):
        """Save trained model"""
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder
        }
        joblib.dump(model_data, Config.DEEP_MODEL_PATH)
        print(f"Model saved to {Config.DEEP_MODEL_PATH}")
    
    def load_model(self):
        """Load trained model"""
        if not os.path.exists(Config.DEEP_MODEL_PATH):
            raise FileNotFoundError("Model not found. Please train the model first.")
        
        model_data = joblib.load(Config.DEEP_MODEL_PATH)
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        print("Model loaded successfully")


class MTCNNFaceDetector:
    """Modern face detection using MTCNN"""
    
    def __init__(self):
        try:
            from mtcnn import MTCNN
            self.detector = MTCNN()
            self.use_mtcnn = True
        except:
            print("MTCNN not available, using Haar Cascade")
            self.detector = cv2.CascadeClassifier(Config.HAAR_CASCADE_PATH)
            self.use_mtcnn = False
    
    def detect_faces(self, image):
        """Detect faces using MTCNN or Haar Cascade"""
        if self.use_mtcnn:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.detector.detect_faces(rgb_image)
            
            faces = []
            for result in results:
                x, y, w, h = result['box']
                # Ensure positive coordinates
                x, y = max(0, x), max(0, y)
                faces.append((x, y, w, h))
            
            return faces
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            return faces
    
    def extract_face(self, image, face_coords):
        """Extract and resize face from image"""
        x, y, w, h = face_coords
        face = image[y:y+h, x:x+w]
        face_resized = cv2.resize(face, Config.FACE_SIZE)
        return face_resized
