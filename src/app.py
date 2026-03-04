"""Modern Face Recognition Attendance System with Deep Learning"""
import cv2
import os
import numpy as np
import joblib
import face_recognition
from flask import Flask, request, render_template, jsonify, redirect, url_for
from src.config import Config
from src.utils import DataManager

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = Config.SECRET_KEY

# Create necessary directories
Config.create_directories()

# Load model if exists
model_data = None
if os.path.exists(Config.DEEP_MODEL_PATH):
    model_data = joblib.load(Config.DEEP_MODEL_PATH)
    print("✅ Deep learning model loaded")
else:
    print("⚠️ No model found. Please train using Jupyter notebook")

# Load background image
try:
    img_background = cv2.imread("FINALBG.jpg")
except:
    img_background = None

@app.route('/')
def home():
    """Home page with attendance dashboard"""
    df = DataManager.get_today_attendance()
    users = DataManager.get_all_users()
    
    return render_template(
        'home.html',
        names=df['Name'].tolist() if not df.empty else [],
        ids=df['ID'].tolist() if not df.empty else [],
        times=df['Time'].tolist() if not df.empty else [],
        l=len(df),
        totalreg=len(users),
        datetoday=Config.DATE_DISPLAY_FORMAT
    )

@app.route('/add_user', methods=['POST'])
def add_user():
    """Add new user to the system"""
    username = request.form.get('newusername')
    user_id = request.form.get('newuserid')
    
    if not username or not user_id:
        return jsonify({'error': 'Name and ID are required'}), 400
    
    # Check if user already exists
    users = DataManager.get_all_users()
    if any(u['id'] == user_id for u in users):
        return jsonify({'error': 'User ID already exists'}), 400
    
    # Capture face images using face_recognition
    cap = cv2.VideoCapture(0)
    captured_count = 0
    frame_count = 0
    
    print(f"Capturing images for {username} (ID: {user_id})")
    
    while captured_count < Config.NUM_IMAGES_PER_USER:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to RGB for face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f'Captured: {captured_count}/{Config.NUM_IMAGES_PER_USER}',
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # Capture every nth frame
            if frame_count % Config.IMAGE_CAPTURE_INTERVAL == 0 and len(face_locations) > 0:
                face_img = frame[top:bottom, left:right]
                face_resized = cv2.resize(face_img, Config.FACE_SIZE)
                DataManager.save_user_face(username, user_id, face_resized, captured_count)
                captured_count += 1
            
            frame_count += 1
        
        cv2.imshow('Adding New User - Press ESC to stop', frame)
        
        if cv2.waitKey(1) == 27 or captured_count >= Config.NUM_IMAGES_PER_USER:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"✅ Captured {captured_count} images")
    print("⚠️ Please retrain the model using the Jupyter notebook")
    
    return redirect(url_for('home'))

@app.route('/take_attendance')
def take_attendance():
    """Take attendance using face recognition"""
    if model_data is None:
        return jsonify({'error': 'No trained model found. Please train using Jupyter notebook.'}), 400
    
    model = model_data['model']
    label_encoder = model_data['label_encoder']
    
    cap = cv2.VideoCapture(0)
    recognized_users = set()
    
    print("Starting attendance capture...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces and get encodings
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            try:
                # Predict
                prediction = model.predict([face_encoding])[0]
                confidence = np.max(model.predict_proba([face_encoding]))
                label = label_encoder.inverse_transform([prediction])[0]
                
                if confidence >= Config.CONFIDENCE_THRESHOLD:
                    name, user_id = label.split('_')
                    
                    # Draw green rectangle for recognized
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                    cv2.putText(
                        frame,
                        f'{name} ({confidence:.2f})',
                        (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )
                    
                    # Mark attendance
                    if label not in recognized_users:
                        if DataManager.save_attendance(name, user_id):
                            recognized_users.add(label)
                            print(f"✅ Attendance marked for {name}")
                else:
                    # Draw red rectangle for unknown
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.putText(
                        frame,
                        'Unknown',
                        (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )
            except Exception as e:
                print(f"Recognition error: {e}")
        
        # Display
        if img_background is not None:
            try:
                img_background[162:162+480, 55:55+640] = cv2.resize(frame, (640, 480))
                cv2.imshow('Attendance System - Press ESC to stop', img_background)
            except:
                cv2.imshow('Attendance System - Press ESC to stop', frame)
        else:
            cv2.imshow('Attendance System - Press ESC to stop', frame)
        
        if cv2.waitKey(1) == 27:  # ESC key
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"✅ Attendance capture completed. {len(recognized_users)} users recognized")
    
    return redirect(url_for('home'))

@app.route('/users')
def get_users():
    """Get all registered users"""
    users = DataManager.get_all_users()
    return jsonify(users)

@app.route('/attendance_history')
def attendance_history():
    """View attendance history"""
    df = DataManager.get_today_attendance()
    return jsonify(df.to_dict('records'))

if __name__ == '__main__':
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG
    )
