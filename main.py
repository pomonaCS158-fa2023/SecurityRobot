import cv2
import dlib
import concurrent.futures
import os
from FaceDatabase import FaceDatabase
from FaceRecognition import FaceRecognition

# Initialize FaceDatabase
face_database = FaceDatabase()
cap = cv2.VideoCapture(1)

# Load door classification model
doorModel = cv2.dnn.readNetFromONNX(os.path.join(os.getcwd(), "Models/doorModel.onnx"))

detector = dlib.get_frontal_face_detector()

# Door state
door_open = False
face_recognition_triggered = False

def process_face(face_crop):
    face_recognition = FaceRecognition()
    label = face_recognition.process_face(face_crop)
    return label

def detect_door_state(frame):
    # TODO: Add door detection logic
    # Return True if door is open, False otherwise
    return True

if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor() as executor:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            capture_text = "Monitoring Door"

            # Detect door state
            if detect_door_state(frame):
                if not door_open:
                    door_open = True
                    face_recognition_triggered = True
            else:
                door_open = False
                face_recognition_triggered = False

            if face_recognition_triggered:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector(gray)

                # Prepare data for processing
                face_data = [(face, frame[face.top():face.bottom(), face.left():face.right()]) for face in faces]

                # Process face embeddings and matching in parallel
                futures = [executor.submit(process_face, face_crop) for _, face_crop in face_data]

                # Process results and update the frame
                for future, (face, _) in zip(concurrent.futures.as_completed(futures), face_data):
                    label = future.result()
                    color = (0,255,0)
                    if label == "INTRUDER":
                        color = (255,0,0)
                    cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), color, 3)
                    cv2.putText(frame, label, (face.left(), face.top()), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                capture_text = "Identifying Faces"

            cv2.imshow(capture_text, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
