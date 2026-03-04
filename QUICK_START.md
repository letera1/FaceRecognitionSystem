# Quick Start Guide

## System is Running! ✅

Server: **http://127.0.0.1:5001**

## Step-by-Step Usage

### Step 1: Add Your First User

1. Open browser: http://127.0.0.1:5001
2. Fill in the form:
   - **Name**: Enter your name (e.g., "John")
   - **ID**: Enter a unique number (e.g., "101")
3. Click **"Add User"** button
4. Webcam will open automatically
5. Look at the camera - system captures 50 images (~10 seconds)
6. Window closes automatically when done

### Step 2: Train the Model

After adding users, run:
```bash
python train_quick.py
```

This will:
- Load all captured face images
- Train KNN model
- Save model to `models/` folder
- Show training accuracy

### Step 3: Take Attendance

1. Go back to http://127.0.0.1:5001
2. Click **"Take Attendance"** button
3. Webcam opens
4. System recognizes faces and shows:
   - Green box = Recognized (with name and confidence)
   - Red box = Unknown
5. Attendance logged automatically
6. Press **ESC** to stop

### Step 4: View Attendance

- Dashboard shows today's attendance
- CSV files saved in `data/attendance/`
- Format: Name, ID, Time, Date

## Current Status

```
✅ Flask server running
✅ Web interface working
✅ Face capture working
⚠️ Model not trained yet (need to add users first)
```

## Workflow

```
Add User → Train Model → Take Attendance → View Records
```

## Troubleshooting

**"No trained model found"**
- You need to add users first
- Then run: `python train_quick.py`

**Camera not working**
- Check camera permissions
- Try different camera: change `cv2.VideoCapture(0)` to `(1)` or `(2)`

**Low accuracy**
- Add more users
- Capture more images per user
- Ensure good lighting

## Files

- `run.py` - Start server
- `train_quick.py` - Train model
- `data/faces/` - User images
- `models/` - Trained models
- `data/attendance/` - Attendance records

## Next Steps

1. Add at least 2-3 users
2. Train the model
3. Test recognition
4. Use for daily attendance
