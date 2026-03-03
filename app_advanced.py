"""Advanced Face Recognition Attendance System"""
import cv2
import os
from flask import Flask, request, render_template, jsonify, redirect, url_for
from config import Config
from models import FaceRecognitionModel
from utils import FaceDetector, DataManager

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = Config.SECRET_KEY

# Create necessary directories
Config.create_directories()

# Initialize components
face_detector = FaceDetector()
face_model = FaceRecognitionModel()

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
        'home_advanced.html',
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
    
    # Capture face images
    cap = cv2.VideoCapture(0)
    captured_count = 0
    frame_count = 0
    
    while captured_count < Config.NUM_IMAGES_PER_USER:
        ret, frame = cap.read()
        if not ret:
            break
        
        faces = face_detector.detect_faces(frame)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
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
            if frame_count % Config.IMAGE_CAPTURE_INTERVAL == 0:
                face_img = face_detector.extract_face(frame, (x, y, w, h))
                DataManager.save_user_face(username, user_id, face_img, captured_count)
                captured_count += 1
            
            frame_count += 1
        
        cv2.imshow('Adding New User', frame)
        
        if cv2.waitKey(1) == 27 or captured_count >= Config.NUM_IMAGES_PER_USER:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Retrain model
    try:
        num_images, num_users = face_model.train(model_type='knn')
        return redirect(url_for('home'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/take_attendance')
def take_attendance():
    """Take attendance using face recognition"""
    if not os.path.exists(Config.MODEL_PATH):
        return jsonify({'error': 'No trained model found. Please add users first.'}), 400
    
    # Load model
    face_model.load_model()
    
    cap = cv2.VideoCapture(0)
    recognized_users = set()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        faces = face_detector.detect_faces(frame)
        
        for (x, y, w, h) in faces:
            face_img = face_detector.extract_face(frame, (x, y, w, h))
            
            try:
                label, confidence = face_model.predict(face_img)
                
                if confidence >= Config.CONFIDENCE_THRESHOLD:
                    name, user_id = label.split('_')
                    
                    # Draw rectangle and label
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y-40), (x+w, y), (0, 255, 0), -1)
                    cv2.putText(
                        frame,
                        f'{name} ({confidence:.2f})',
                        (x+5, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )
                    
                    # Mark attendance
                    if label not in recognized_users:
                        if DataManager.save_attendance(name, user_id):
                            recognized_users.add(label)
                else:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(
                        frame,
                        'Unknown',
                        (x+5, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )
            except Exception as e:
                print(f"Recognition error: {e}")
        
        # Display with background if available
        if img_background is not None:
            try:
                img_background[162:162+480, 55:55+640] = cv2.resize(frame, (640, 480))
                cv2.imshow('Attendance System', img_background)
            except:
                cv2.imshow('Attendance System', frame)
        else:
            cv2.imshow('Attendance System', frame)
        
        if cv2.waitKey(1) == 27:  # ESC key
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
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
