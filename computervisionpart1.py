import cv2

# Load model face detection (Haar Cascade Classifier)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Opening Webcam
cap = cv2.VideoCapture(0)
