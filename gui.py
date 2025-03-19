from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
 

import h5py
import tkinter as tk
import os
from collections import deque  # For smoothing predictions

# Enable GPU memory growth for TensorFlow
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# Define emotion model
emotion_model = Sequential([
    Input(shape=(48, 48, 1)),
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

# Load the emotion model weights
weights_path = r"D:\facial recognition\emotion_model.weights.h5"
if os.path.exists(weights_path):
    with h5py.File(weights_path, 'r') as f:
        if len(f.keys()) > 0:  # Check if file is not empty
            emotion_model.load_weights(weights_path)
        else:
            print("Error: Weight file is empty or corrupted.")
            exit()
else:
    print("Error: Weight file not found.")
    exit()

# Emotion dictionary
emotion_dict = {
    0: "Angry", 1: "Disgusted", 2: "Fearful",
    3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"
}

# Define paths for the face detection model
prototxt_path = r"D:\facial recognition\deploy.prototxt"
model_path = r"D:\facial recognition\res10_300x300_ssd_iter_140000.caffemodel"

if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
    print(f"Error: Missing face detection model files in {os.path.dirname(prototxt_path)}.")
    exit()

# Load face detection model
face_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Open webcam
cap1 = cv2.VideoCapture(0)
if not cap1.isOpened():
    print("Error: Could not open camera.")
    exit()

# Tkinter GUI setup
root = tk.Tk()
root.title("Real-Time Emotion Detection")
root.geometry("800x600")
root['bg'] = 'black'

lmain = tk.Label(root, padx=50, bd=10)
lmain.pack()

# Queue to store last 5 predictions for smoothing
pred_queue = deque(maxlen=5)

def smooth_predictions(new_label):
    pred_queue.append(new_label)
    return max(set(pred_queue), key=pred_queue.count)  # Most common label

def detect_faces(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:  # Increased threshold from 0.5 to 0.7
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            faces.append(box.astype("int"))
    print(f"Detected {len(faces)} face(s)")
    return faces

def show_video():
    ret, frame = cap1.read()
    if not ret:
        return

    frame = cv2.resize(frame, (500, 400))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detect_faces(frame)

    for (x, y, x1, y1) in faces:
        roi_gray = gray_frame[y:y1, x:x1]
        if roi_gray.size == 0:
            continue
        
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)) / 255.0, -1), 0)
        prediction = emotion_model.predict(cropped_img)
        
        max_index = int(np.argmax(prediction))
        confidence = prediction[0][max_index]
        label = emotion_dict[max_index]
        
        if label == "Fearful" and confidence < 0.7:
            label = "Neutral"
        
        label = smooth_predictions(label)
        print(f"Predictions: {prediction[0]} | Selected: {label} ({confidence:.2f})")

        cv2.rectangle(frame, (x, y), (x1, y1), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    root.after(10, show_video)

show_video()
root.mainloop()
cap1.release()
cv2.destroyAllWindows()
