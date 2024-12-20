import cv2

# Load model face detection (Haar Cascade Classifier)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Opening Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Tidak dapat membuka kamera.")
    exit()

while True:
    # reading the frame
    ret, frame = cap.read()

    if not ret:
        print("Gagal membaca frame dari kamera.")
        break
    # Changin read image into grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # reading the face throughout the webcam
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # draw a square around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # showing the frame
    cv2.imshow('Face Detection', frame)

    # shortcut for quit the program (using q)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# running and closing the program
cap.release()
cv2.destroyAllWindows()
