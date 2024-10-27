import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('path_to_trained_model.h5')

# Define OpenCV to capture from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray, (IMG_WIDTH, IMG_HEIGHT))
    input_frame = np.expand_dims(resized_frame, axis=[0, -1]) / 255.0

    # Make prediction
    prediction = model.predict(input_frame)
    if prediction > 0.5:
        label = 'Drowsy'
    else:
        label = 'Alert'

    # Display the label on the frame
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the frame with the prediction
    cv2.imshow('Driver Drowsiness Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
