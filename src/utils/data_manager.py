"""Data management utilities"""
import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from src.config import Config

class DataManager:
    @staticmethod
    def save_user_face(username, user_id, face_image, image_index):
        """Save a user's face image"""
        user_folder = os.path.join(Config.FACES_DIR, f"{username}_{user_id}")
        os.makedirs(user_folder, exist_ok=True)
        
        filename = f"{username}_{image_index}.jpg"
        filepath = os.path.join(user_folder, filename)
        cv2.imwrite(filepath, face_image)
        return filepath
    
    @staticmethod
    def load_all_faces():
        """Load all face images and labels"""
        faces = []
        labels = []
        
        if not os.path.exists(Config.FACES_DIR):
            return np.array([]), []
        
        for user_folder in os.listdir(Config.FACES_DIR):
            user_path = os.path.join(Config.FACES_DIR, user_folder)
            if not os.path.isdir(user_path):
                continue
                
            for img_name in os.listdir(user_path):
                img_path = os.path.join(user_path, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    faces.append(img)
                    labels.append(user_folder)
        
        return np.array(faces), labels
    
    @staticmethod
    def get_all_users():
        """Get list of all registered users"""
        if not os.path.exists(Config.FACES_DIR):
            return []
        
        users = []
        for user_folder in os.listdir(Config.FACES_DIR):
            if os.path.isdir(os.path.join(Config.FACES_DIR, user_folder)):
                name, user_id = user_folder.split('_')
                users.append({'name': name, 'id': user_id, 'folder': user_folder})
        return users
    
    @staticmethod
    def save_attendance(name, user_id):
        """Save attendance record"""
        attendance_file = os.path.join(
            Config.ATTENDANCE_DIR, 
            f"Attendance-{Config.DATE_FORMAT}.csv"
        )
        
        # Create file with headers if it doesn't exist
        if not os.path.exists(attendance_file):
            df = pd.DataFrame(columns=['Name', 'ID', 'Time', 'Date'])
            df.to_csv(attendance_file, index=False)
        
        # Read existing attendance
        df = pd.read_csv(attendance_file)
        
        # Check if user already marked attendance today
        if int(user_id) not in df['ID'].values:
            current_time = datetime.now().strftime("%H:%M:%S")
            new_record = pd.DataFrame({
                'Name': [name],
                'ID': [user_id],
                'Time': [current_time],
                'Date': [Config.DATE_DISPLAY_FORMAT]
            })
            df = pd.concat([df, new_record], ignore_index=True)
            df.to_csv(attendance_file, index=False)
            return True
        return False
    
    @staticmethod
    def get_today_attendance():
        """Get today's attendance records"""
        attendance_file = os.path.join(
            Config.ATTENDANCE_DIR,
            f"Attendance-{Config.DATE_FORMAT}.csv"
        )
        
        if os.path.exists(attendance_file):
            df = pd.read_csv(attendance_file)
            return df
        return pd.DataFrame(columns=['Name', 'ID', 'Time', 'Date'])
