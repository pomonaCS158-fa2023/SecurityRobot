import cv2
import os
import numpy as np
from FaceDatabase import FaceDatabase
from FaceRecognition import FaceRecognition
from sklearn.cluster import KMeans

acceptable_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

def is_image_file(filename):
    return os.path.splitext(filename)[1].lower() in acceptable_extensions

def remove_outliers(embeddings, threshold=2.0):
    mean_embedding = np.mean(embeddings, axis=0)
    distances = np.linalg.norm(embeddings - mean_embedding, axis=1)
    return embeddings[distances < threshold]

face_database = FaceDatabase()
face_recognition = FaceRecognition()
faces_dir = "Faces"

for person_name in os.listdir(faces_dir):
    person_dir = os.path.join(faces_dir, person_name)
    if not os.path.isdir(person_dir):
        continue

    embeddings = []
    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_name)
        if is_image_file(image_path):
            face_image = cv2.imread(image_path)
            embedding = face_recognition.get_face_embedding(face_image)
            if embedding is not None:
                embeddings.append(embedding)

    if embeddings:
        embeddings = np.array(embeddings)
        embeddings = remove_outliers(embeddings)

        # Clustering embeddings and finding the largest cluster
        kmeans = KMeans(n_clusters=min(5, len(embeddings)), random_state=0).fit(embeddings)
        largest_cluster = np.argmax(np.bincount(kmeans.labels_))
        cluster_embeddings = embeddings[kmeans.labels_ == largest_cluster]

        average_embedding = np.mean(cluster_embeddings, axis=0)
        face_database.add_face(person_name, average_embedding)
    else:
        print(f"No embeddings generated for {person_name}.")