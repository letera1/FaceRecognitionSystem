"""Create demo user with sample images"""
import sys
import os
import cv2
import numpy as np

sys.path.insert(0, '.')
from src.config import Config
from src.utils import DataManager

def create_demo_user():
    """Create a demo user with generated face images"""
    print("Creating demo user...")
    
    # Create demo user folder
    demo_name = "Demo"
    demo_id = "999"
    user_folder = os.path.join(Config.FACES_DIR, f"{demo_name}_{demo_id}")
    os.makedirs(user_folder, exist_ok=True)
    
    # Generate 50 sample face images (simple colored rectangles as placeholders)
    for i in range(50):
        # Create a simple image (this is just for demo - replace with real webcam capture)
        img = np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)
        
        # Add some variation
        cv2.circle(img, (64, 64), 30, (255, 200, 150), -1)  # Face-like circle
        cv2.circle(img, (50, 50), 5, (0, 0, 0), -1)  # Eye
        cv2.circle(img, (78, 50), 5, (0, 0, 0), -1)  # Eye
        cv2.ellipse(img, (64, 75), (15, 10), 0, 0, 180, (0, 0, 0), 2)  # Mouth
        
        filename = os.path.join(user_folder, f"{demo_name}_{i}.jpg")
        cv2.imwrite(filename, img)
    
    print(f"✅ Created demo user: {demo_name} (ID: {demo_id})")
    print(f"   Location: {user_folder}")
    print(f"   Images: 50")
    
    return True

if __name__ == '__main__':
    create_demo_user()
    print("\nNow run: python train_quick.py")
