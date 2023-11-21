import pickle
import os

"""
    A class for managing a face database using Dlib.

    It stores a dictionary in the Face Database folder
"""
class FaceDatabase:
    def __init__(self, database_filename="face_data.pkl"):
        # Find the correct path in the directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.database_path = os.path.join(script_dir, "FaceData", database_filename)

        # Ensure the directory for the database file exists
        os.makedirs(os.path.dirname(self.database_path), exist_ok=True)

        # Load or initialize the database
        self.database = self.__load_database()

    def __load_database(self):
        if os.path.exists(self.database_path):
            with open(self.database_path, 'rb') as file:
                return pickle.load(file)
        else:
            return {}
        
    def get_data(self):
        return self.database.items()

    def save_database(self):
        with open(self.database_path, 'wb') as file:
            print(self.database_path)
            pickle.dump(self.database, file)

    def add_face(self, name, embedding):
        if embedding is not None:
            self.database[name] = embedding
            self.save_database()
