import cv2
import numpy as np
import dlib
from FaceDatabase import FaceDatabase

class FaceRecognition:

    @staticmethod
    def process_face(face_crop):
        """
        Process a single cropped face image: get embeddings, check if the face is a match.
        Returns the label (name or "INTRUDER").
        """

        # Initialize the models in each process to allow for parallel processing
        predictor = dlib.shape_predictor("Models/shape_predictor_68_face_landmarks.dat")
        facerec = dlib.face_recognition_model_v1("Models/dlib_face_recognition_resnet_model_v1.dat")
        detector = dlib.get_frontal_face_detector()

        # Extract the embeddings from the face using the models
        face_image_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

        detected_faces = detector(face_image_rgb, 1)
        if len(detected_faces) == 0:
            return "Analyzing"

        face_rect = detected_faces[0]  # This is the rectangle
        landmarks = predictor(face_image_rgb, face_rect)
        embedding = np.array(facerec.compute_face_descriptor(face_image_rgb, landmarks))

        # Check if the face is a match
        return FaceRecognition.match_face(embedding)

    @staticmethod
    def get_face_embedding(image):

        # Initialize the models in each process to allow for parallel processing
        predictor = dlib.shape_predictor("Models/shape_predictor_68_face_landmarks.dat")
        facerec = dlib.face_recognition_model_v1("Models/dlib_face_recognition_resnet_model_v1.dat")
        detector = dlib.get_frontal_face_detector()

        # Extract the embeddings from the face using the models
        face_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        detected_faces = detector(face_image_rgb, 1)
        if len(detected_faces) == 0:
            return "No Face Detected"

        face_rect = detected_faces[0]
        landmarks = predictor(face_image_rgb, face_rect)
        embedding = np.array(facerec.compute_face_descriptor(face_image_rgb, landmarks))

        return embedding
    
    @staticmethod
    def match_face(embedding):
        face_database = FaceDatabase()
        for name, db_embedding in face_database.get_data():
            dist = np.linalg.norm(db_embedding - embedding)
            if dist < 0.2:
                return name  # Authorized face
        return "INTRUDER"  # Unauthorized or unknown face
