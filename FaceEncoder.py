import cv2
import os
import numpy as np
from FaceDatabase import FaceDatabase
from FaceRecognition import FaceRecognition


import os

acceptable_extensions = {'.jpg', '.jpe' '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

def is_image_file(filename):
    return os.path.splitext(filename)[1].lower() in acceptable_extensions

# Initialize FaceDatabase and FaceRecognition
face_database = FaceDatabase()
face_recognition = FaceRecognition()

# Directory containing folders of faces
faces_dir = "Faces"

# Iterate through each person's folder
for person_name in os.listdir(faces_dir):
    person_dir = os.path.join(faces_dir, person_name)
    
    # Skip if not a directory
    if not os.path.isdir(person_dir):
        continue

    # Process images and create embeddings
    embeddings = []
    for image_name in os.listdir(person_dir):
        print(image_name)
        image_path = os.path.join(person_dir, image_name)
        if is_image_file(image_path):
            face_image = cv2.imread(image_path)
            embedding = face_recognition.get_face_embedding(face_image)
            if embedding is not None:
                embeddings.append(embedding)

    # Create an average embedding if there are multiple images
    if embeddings:
        average_embedding = np.mean(embeddings, axis=0)
        face_database.add_face(person_name, average_embedding)
    else:
        print(f"No embeddings generated for {person_name}.")
