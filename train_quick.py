"""Quick training script for face recognition model"""
import sys
import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Add src to path
sys.path.insert(0, '.')
from src.config import Config
from src.utils import DataManager

def train_model():
    """Train face recognition model from existing data"""
    print("=" * 60)
    print("QUICK TRAINING SCRIPT")
    print("=" * 60)
    
    # Load all faces
    faces, labels = DataManager.load_all_faces()
    
    if len(faces) == 0:
        print("\n❌ No training data found!")
        print("\nPlease add users first:")
        print("1. Go to http://127.0.0.1:5001")
        print("2. Fill in name and ID")
        print("3. Click 'Add User'")
        print("4. Let system capture 50 images")
        print("5. Run this script again")
        return False
    
    print(f"\n✅ Found {len(faces)} images from {len(set(labels))} users")
    print(f"Users: {set(labels)}")
    
    # Preprocess faces
    print("\nPreprocessing images...")
    processed_faces = []
    
    for face in faces:
        # Convert to grayscale
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        # Normalize
        normalized = gray / 255.0
        # Flatten
        processed_faces.append(normalized.ravel())
    
    X = np.array(processed_faces)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    
    print(f"Feature shape: {X.shape}")
    
    # Train KNN model
    print("\nTraining KNN model...")
    model = KNeighborsClassifier(
        n_neighbors=min(5, len(set(labels))),
        weights='distance',
        metric='euclidean'
    )
    
    model.fit(X, y)
    
    # Calculate accuracy
    train_accuracy = model.score(X, y)
    print(f"Training accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    
    # Save model
    model_data = {
        'model': model,
        'label_encoder': label_encoder,
        'num_classes': len(set(labels)),
        'num_samples': len(X)
    }
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model_data, Config.DEEP_MODEL_PATH)
    
    print(f"\n✅ Model saved to: {Config.DEEP_MODEL_PATH}")
    print(f"   - Model type: KNN")
    print(f"   - Number of users: {len(set(labels))}")
    print(f"   - Training samples: {len(X)}")
    print(f"   - Accuracy: {train_accuracy:.4f}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print("\nYou can now:")
    print("1. Restart the Flask app (or it will auto-reload)")
    print("2. Click 'Take Attendance'")
    print("3. System will recognize faces")
    
    return True

if __name__ == '__main__':
    success = train_model()
    sys.exit(0 if success else 1)
