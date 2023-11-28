import cv2
import dlib
import concurrent.futures
import os
from FaceDatabase import FaceDatabase
from FaceRecognition import FaceRecognition
import torch
from torchvision import transforms
from PIL import Image

# Greet the authorized persons
def greet():
    pass

def process_face(face_crop):
    face_recognition = FaceRecognition()
    label = face_recognition.process_face(face_crop)
    return label

# Used to transform photos for door model
transformer = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])
def preprocess_frame_for_door_model(frame):
    # Convert OpenCV BGR image to RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    # Apply the transforms
    transformed_frame = transformer(pil_image)
    
    return transformed_frame

def detect_door_state(frame):
    frame = preprocess_frame_for_door_model(frame).numpy().transpose((1,2,0))

    # Run the model
    blob = cv2.dnn.blobFromImage(frame)
    doorModel.setInput(blob)
    output = doorModel.forward()

    # Return True if door is open, False otherwise
    return output[0][0] < 0.5

if __name__ == '__main__':

    # Initialize FaceDatabase
    face_database = FaceDatabase()
    cap = cv2.VideoCapture(0)

    # Load door classification model
    doorModel = cv2.dnn.readNetFromONNX(os.path.join(os.getcwd(), "Models/doorModel.onnx"))

    detector = dlib.get_frontal_face_detector()

    # Door state
    door_open = False
    face_recognition_triggered = False
    
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

                # Find faces
                faces = detector(gray)
                face_data = []
                for face in faces:
                    face_crop = frame[face.top():face.bottom(), face.left():face.right()]
                    if face_crop.size != 0:
                        face_data.append((face, face_crop))
                
                
                # Process face embeddings and matching in parallel
                futures = [executor.submit(process_face, face_crop) for _, face_crop in face_data]

                intrusion = False
                authorization_granted = False
                # Process results and update the frame
                for future, (face, _) in zip(concurrent.futures.as_completed(futures), face_data):
                    label = future.result()
                    color = (0,255,0)

                    if label == "INTRUDER":
                        if authorization_granted == False:
                            intrusion = True
                        color = (0,0,255)
                    elif label == "Analyzing":
                        color = (255,0,0)
                    else:
                        authorization_granted = True
                    cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), color, 3)
                    cv2.putText(frame, label, (face.left(), face.top()), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                capture_text = "Identifying Faces"

                if authorization_granted:
                    greet()
                    # Wait for door close and then open 
                if intrusion:
                    # Ask to leave
                    pass

            cv2.imshow(capture_text, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
