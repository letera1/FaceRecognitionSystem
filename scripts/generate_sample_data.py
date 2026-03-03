"""Generate sample training data for testing"""
import cv2
import os
import sys
sys.path.append('..')

from config import Config
from utils import DataManager

Config.create_directories()

def generate_sample_faces():
    """Generate sample face data using webcam"""
    print("Sample Data Generator")
    print("=" * 50)
    
    sample_users = []
    
    while True:
        name = input("\nEnter user name (or 'done' to finish): ").strip()
        if name.lower() == 'done':
            break
        
        user_id = input("Enter user ID: ").strip()
        
        if not name or not user_id:
            print("Invalid input. Try again.")
            continue
        
        sample_users.append((name, user_id))
    
    if not sample_users:
        print("No users to add.")
        return
    
    # Initialize face detector
    face_cascade = cv2.CascadeClassifier(Config.HAAR_CASCADE_PATH)
    cap = cv2.VideoCapture(0)
    
    for name, user_id in sample_users:
        print(f"\n\nCapturing images for {name} (ID: {user_id})")
        print("Press SPACE to capture, ESC to skip")
        
        captured = 0
        frame_count = 0
        
        while captured < Config.NUM_IMAGES_PER_USER:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f'{name}: {captured}/{Config.NUM_IMAGES_PER_USER}',
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # Auto-capture every 5 frames
                if frame_count % 5 == 0 and len(faces) > 0:
                    face_img = cv2.resize(frame[y:y+h, x:x+w], Config.FACE_SIZE)
                    DataManager.save_user_face(name, user_id, face_img, captured)
                    captured += 1
                    print(f"Captured {captured}/{Config.NUM_IMAGES_PER_USER}")
                
                frame_count += 1
            
            cv2.imshow('Capture Faces', frame)
            
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break
            elif captured >= Config.NUM_IMAGES_PER_USER:
                break
        
        print(f"Completed: {captured} images captured for {name}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 50)
    print("Sample data generation complete!")
    print(f"Data saved in: {Config.FACES_DIR}")

if __name__ == '__main__':
    generate_sample_faces()
