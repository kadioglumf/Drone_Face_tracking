import face_recognition
import cv2
import numpy as np
import sys

class FaceRecogntion(object):
    def __init__(self):

        self.fatih_image = face_recognition.load_image_file("fatih.jpg")
        self.fatih_face_encoding = face_recognition.face_encodings(self.fatih_image)[0]

        self.known_face_encodings = [
            self.fatih_face_encoding
        ]
        self.known_face_names = [
            "Fatih"
        ]

        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_this_frame = True   


    def recognition(self,frame):
    


        # Daha hızlı yüz tanıma işlemi için videonun karesini 1/4 boyutuna gitirildi
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # opencv'de kullanılan BGR görüntünün face_recognition da kullanılan RGB' ye dönüşümü
        rgb_small_frame = small_frame[:, :, ::-1]


        if self.process_this_frame:
            # Geçerli yüz karesinde tüm yüzleri ve yüz kodlarını bulma işlemei
            self.face_locations = face_recognition.face_locations(rgb_small_frame)
            self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

            self.face_names = []
            for face_encoding in self.face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

                self.face_names.append(name)

        self.process_this_frame = not self.process_this_frame

        return self.face_locations, self.face_names


    
