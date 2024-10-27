import cv2
import dlib
import numpy as np
from tensorflow.keras.models import load_model
from imutils import face_utils
from scipy.spatial import distance as dist
import pygame

# Load the trained model
model = load_model('cnn_eye_state.h5')

# Initialize face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Function to compute Eye Aspect Ratio (EAR)
def calculate_EAR(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Initialize Pygame for the alarm
pygame.mixer.init()
pygame.mixer.music.load('alarm.wav')

# EAR threshold for drowsiness detection
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 20

# Counter to track consecutive frames where eyes are closed
counter = 0

# Start video stream from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = detector(gray)
    
    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # Extract the left and right eye coordinates
        leftEye = shape[face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][0]:
                        face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][1]]
        rightEye = shape[face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][0]:
                         face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][1]]

        # Preprocess the eye region and predict eye state (open/closed) using the CNN model
        leftEyeImg = cv2.resize(gray[min(leftEye[:, 1]):max(leftEye[:, 1]),
                                      min(leftEye[:, 0]):max(leftEye[:, 0])], (24, 24))
        rightEyeImg = cv2.resize(gray[min(rightEye[:, 1]):max(rightEye[:, 1]),
                                      min(rightEye[:, 0]):max(rightEye[:, 0])], (24, 24))

        leftEyeState = model.predict(leftEyeImg.reshape(1, 24, 24, 1) / 255.0)
        rightEyeState = model.predict(rightEyeImg.reshape(1, 24, 24, 1) / 255.0)

        # Check if both eyes are classified as closed
        if leftEyeState < 0.5 and rightEyeState < 0.5:
            counter += 1
        else:
            counter = 0

        # If eyes are closed for enough frames, trigger the alarm
        if counter >= EYE_AR_CONSEC_FRAMES:
            pygame.mixer.music.play()
        else:
            pygame.mixer.music.stop()

        # Draw the eyes on the frame
        cv2.polylines(frame, [leftEye], True, (0, 255, 0), 1)
        cv2.polylines(frame, [rightEye], True, (0, 255, 0), 1)

    # Display the resulting frame
    cv2.imshow('Drowsiness Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
