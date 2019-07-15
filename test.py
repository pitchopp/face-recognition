import tkinter
import face_recognition
import cv2
import numpy as np
import os
import PIL.Image
import PIL.ImageTk
from datetime import datetime
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from gtts import gTTS

from model import get_model


class App:
    def __init__(self, window, window_title, video_source=0):
        self.photo = None
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width=self.vid.width, height=self.vid.height)
        self.canvas.pack()

        self.label_snapshot = tkinter.Label(
            window,
            text="Entrez votre Prenom Nom avant de cliquer sur le bouton d'ajout.\n"
                 "Assurez-vous de regarder la caméra et que votre visage est bien détecté(apparition du carré bleu)")
        self.label_snapshot.pack()

        # Text that will be used as name for the snapshot
        self.txt_snapshot = tkinter.Entry(window)
        self.txt_snapshot.pack()
        self.txt_snapshot.focus_set()

        # Button that lets the user take a snapshot
        self.btn_snapshot = tkinter.Button(window, text="Ajouter à la Base de Données", width=50, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()

        self.window.mainloop()

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_raw_frame()

        if ret:
            file_name = self.txt_snapshot.get()
            if file_name not in [None, '']:
                cv2.imwrite("known-faces/" + file_name + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                self.vid.reload_known_faces()

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

        self.window.after(self.delay, self.update)


class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 2000)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1400)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)
        self.known_face_encodings = []
        self.known_face_names = []
        self.reload_known_faces()

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.img_resize = 3

        # Load Emotion model
        self.emotion_model = load_model('./models/emotion_model.hdf5')
        self.emotion_classes = [u"Colere", u"Degout", u"Peur", u"Joie", u"Tristesse", u"Etonnement", u"Neutre"]

        # Load Gender model
        self.gender_model = load_model('./models/gender_detection.model')
        self.gender_classes = ['Homme', 'Femme']

        self.age_model = get_model()
        self.age_model.load_weights('./models/age_model.hdf5')

        self.last_seen = {}

    def reload_known_faces(self):
        # load known faces
        known_faces = []
        try:
            for img in os.listdir('known-faces'):
                if img.endswith('jpeg') or img.endswith('jpg') or img.endswith('png'):
                    tmp = face_recognition.face_encodings(face_recognition.load_image_file('known-faces/' + img))[0]
                    known_faces.append((img.split('.')[0], tmp))
        except IndexError:
            pass
        self.known_face_encodings = [x[1] for x in known_faces]
        self.known_face_names = [x[0] for x in known_faces]

    def get_raw_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            return None, None

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.get_raw_frame()
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame.copy(), (0, 0), fx=1/self.img_resize, fy=1/self.img_resize)
            # small_frame = frame.copy()

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame.copy()[:, :, ::-1]

            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            face_emotions = []
            face_genders = []
            face_ages = []
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                top *= self.img_resize
                right *= self.img_resize
                bottom *= self.img_resize
                left *= self.img_resize
                # See if the face is a match for the known face(s)

                face_names.append(self.get_name(face_encoding))

                # Emotion
                face_frame = frame[top:bottom, left:right]
                face_emotions.append(self.get_emotion(face_frame))

                face_genders.append(self.get_gender(face_frame))
                face_ages.append(self.get_age(face_frame))

            # say hello
            self.say_hello(face_names)

            # Display the results
            for (top, right, bottom, left), name, emotion, gender, age in zip(face_locations, face_names, face_emotions, face_genders, face_ages):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= self.img_resize
                right *= self.img_resize
                bottom *= self.img_resize
                left *= self.img_resize

                # Draw a label with a name below the face
                if gender == 'Homme':
                    gender_color = (0, 0, 255)
                elif gender == 'Femme':
                    gender_color = (255, 0, 0)
                else:
                    gender_color = (0, 255, 0)

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), gender_color, 2)

                cv2.rectangle(frame, (left, bottom), (right, bottom + 40), gender_color, cv2.FILLED)
                font = int(cv2.FONT_HERSHEY_DUPLEX / 2)
                cv2.putText(frame, name + ', ' + str(int(age)), (left + 6, bottom + 20), font, 1.0, (255, 255, 255), 1)
                cv2.putText(frame, gender + ', ' + emotion, (left + 6, bottom + 36), font, 1.0, (255, 255, 255), 1)
            return ret, frame
        else:
            return None, None

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

    def get_emotion(self, frame):
        try:
            roi = frame.copy()
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            preds = self.emotion_model.predict(roi)[0]
            return self.emotion_classes[preds.argmax()]
        except Exception:
            return "Inconnu"

    def get_name(self, face_encoding):
        name = "Invité"
        try:
            if len(self.known_face_encodings) > 0:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)

                # use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
        except Exception:
            pass
        return name

    def get_gender(self, frame):
        try:
            # preprocessing for gender detection model
            face_crop = frame.copy()
            face_crop = cv2.resize(face_crop, (96, 96))
            face_crop = face_crop.astype("float") / 255.0
            face_crop = img_to_array(face_crop)
            face_crop = np.expand_dims(face_crop, axis=0)

            # apply gender detection on face
            conf = self.gender_model.predict(face_crop)[0]

            # get label with max accuracy
            idx = np.argmax(conf)

            return self.gender_classes[idx]
        except Exception:
            return "Unknown"

    def get_age(self, frame):
        try:
            # preprocessing for gender detection model
            img_size = self.age_model.input.shape.as_list()[1]
            img = frame.copy()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tmp = np.empty((1, img_size, img_size, 3))
            tmp[0, :, :, :] = cv2.resize(img, (img_size, img_size))
            prediction = self.age_model.predict(tmp)
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = prediction.dot(ages).flatten()
            return predicted_ages[0]
        except Exception:
            return 0

    def say_hello(self, face_names):
        # pass
        names = []
        for face_name in face_names:
            last_seen = self.last_seen.get(face_name.lower().strip())
            if last_seen is None or last_seen.date() < datetime.today().date():
                names.append(face_name)
            self.last_seen[face_name.lower().strip()] = datetime.now()
        if len(names) > 0:
            text = 'Bonjour ' + ' et '.join(names)
            language = 'fr'
            myobj = gTTS(text=text, lang=language, slow=False)
            myobj.save('welcome.mp3')
            os.system("mpg321 welcome.mp3")


# Create a window and pass it to the Application object
App(tkinter.Tk(), "Bienvenue chez Visian", video_source=2)
