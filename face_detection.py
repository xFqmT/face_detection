import cv2
import os
import time
from datetime import datetime

def initialize_face_detector():
    """
    Initialize the face detection model (Haar Cascade).
    Returns the classifier object.
    """
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    
    if not os.path.isfile(face_cascade_path):
        raise FileNotFoundError(f"Haar Cascade file not found at: {face_cascade_path}")
    
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    return face_cascade

def initialize_webcam(camera_index=0):
    """
    Initialize the webcam capture.
    Returns the video capture object.
    """
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        raise RuntimeError("Could not initialize webcam. Please check your camera connection.")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    return cap

def detect_faces(frame, face_cascade):
    """
    Detect faces in a frame using the provided classifier.
    Returns the frame with bounding boxes drawn around faces and the list of face rectangles.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return frame, faces

def save_screenshot(frame, output_dir="screenshots"):
    """
    Save the current frame as an image file with a timestamp.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"screenshot_{timestamp}.png")
    cv2.imwrite(filename, frame)
    print(f"Screenshot saved as {filename}")

def save_detected_faces(frame, faces, output_dir="detected_faces"):
    """
    Save each detected face as a separate image file.
    """
    if len(faces) == 0:
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i, (x, y, w, h) in enumerate(faces):
        face_img = frame[y:y+h, x:x+w]
        filename = os.path.join(output_dir, f"face_{timestamp}_{i}.png")
        cv2.imwrite(filename, face_img)
        print(f"Face image saved as {filename}")

def main():
    print("Initializing face detection application...")
    
    try:
        face_cascade = initialize_face_detector()
        cap = initialize_webcam()
        
        print("Press 'q' to quit, 's' to save screenshot, 'f' to save detected faces")
        
        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)  
            
            processed_frame, faces = detect_faces(frame, face_cascade)
            
            cv2.imshow('Face Detection', processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  
                break
            elif key == ord('s'): 
                save_screenshot(processed_frame)
            elif key == ord('f') and len(faces) > 0:  
                save_detected_faces(processed_frame, faces)

            time.sleep(0.03)
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        print("Application closed.")

if __name__ == "__main__":
    main()