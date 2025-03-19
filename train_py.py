# -*- coding: utf-8 -*-
"""train.py

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1anw4XFKOOWQQA7wAUr_IWrBZ1YFA71kN
"""

# Install p7zip
!apt-get install p7zip-full

# Extract the archive using 7zip
!7z x /content/archive.zip -o/content/dataset

import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import tensorflow as tf # Import tensorflow

# Initialize directories and data generators
# Use os.path.join to create platform-independent paths
train_dir = os.path.join('/content/dataset', 'train')
val_dir = os.path.join('/content/dataset', 'test')
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical',
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical',
    shuffle=False
)

# Build the emotion detection model
emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

# Load pre-trained weights (uncomment if needed)
# emotion_model.load_weights('emotion_model.h5')

# Disable OpenCL for compatibility
cv2.ocl.setUseOpenCL(False)

# Emotion dictionary
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Compile the model
# Compile the model
emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001, decay=1e-6), metrics=['accuracy'])

# Train the model
emotion_model_info = emotion_model.fit(
    train_generator,  # Repeat the dataset indefinitely
    steps_per_epoch=len(train_generator),
    epochs=20,
    validation_data=validation_generator,  # Repeat validation data as well
    validation_steps=len(validation_generator)
)

# Save the model weights
emotion_model.save_weights('emotion_model.weights.h5')


# Initialize webcam and Haar Cascade classifier
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Load Haar Cascade for face detection
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
bounding_box = cv2.CascadeClassifier(face_cascade_path)

if bounding_box.empty():
    print("Error: Could not load Haar Cascade classifier.")
    exit()

# Start webcam feed
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        # Draw bounding box around the face
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)

        # Extract the region of interest (ROI) and preprocess it
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # Predict the emotion
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))

        # Display the predicted emotion
        cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Video', cv2.resize(frame, (1200, 860), interpolation=cv2.INTER_CUBIC))

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

