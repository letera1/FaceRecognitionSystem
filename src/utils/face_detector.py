"""Face detection utilities"""
import cv2
import numpy as np
from src.config import Config

class FaceDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(Config.HAAR_CASCADE_PATH)
        
    def detect_faces(self, image):
        """Detect faces in an image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
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
    
    def preprocess_face(self, face):
        """Preprocess face for model input"""
        # Convert to grayscale
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        # Histogram equalization for better contrast
        equalized = cv2.equalizeHist(gray)
        # Normalize
        normalized = equalized / 255.0
        return normalized
