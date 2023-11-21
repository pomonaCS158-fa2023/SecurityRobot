import cv2
import dlib
import concurrent.futures
from FaceDatabase import FaceDatabase
from FaceRecognition import FaceRecognition

def process_face(face_crop):
        face_recognition = FaceRecognition()
        label = face_recognition.process_face(face_crop)
        return label

# Initialize FaceDatabase
face_database = FaceDatabase()

cap = cv2.VideoCapture(1)
detector = dlib.get_frontal_face_detector()

if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor() as executor:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

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

            cv2.imshow('Real-time Face Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()