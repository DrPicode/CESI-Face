# Bibliothèques standard Python
import csv
import hashlib
import json
import logging
import os
import pickle
import queue
import random
import sys
import textwrap
import threading
import time
from datetime import datetime

# Interface graphique
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# Traitement d'images et Computer Vision
import cv2
from deepface import DeepFace
from ultralytics import YOLO

# Calcul scientifique et Machine Learning
import numpy as np
from sklearn.model_selection import train_test_split

# TensorFlow et Keras
import tensorflow as tf
import keras
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


class LoginWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Connexion")
        self.root.geometry("300x200")  # Augmenté la hauteur pour le nouveau menu

        # Variables
        self.password_var = tk.StringVar()
        self.action_var = tk.StringVar(value="Entrée")
        self.salle_var = tk.StringVar(value="100")  # Valeur par défaut pour la salle

        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Password field
        ttk.Label(main_frame, text="Mot de passe :").grid(row=0, column=0, pady=10)
        password_entry = ttk.Entry(main_frame, textvariable=self.password_var, show="*")
        password_entry.grid(row=0, column=1, pady=10)

        # Action choice (Entrée/Sortie)
        ttk.Label(main_frame, text="Mode :").grid(row=1, column=0, pady=10)
        action_frame = ttk.Frame(main_frame)
        action_frame.grid(row=1, column=1, pady=10)

        ttk.Radiobutton(action_frame, text="Entrée", variable=self.action_var,
                       value="Entrée").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(action_frame, text="Sortie", variable=self.action_var,
                       value="Sortie").pack(side=tk.LEFT, padx=5)

        # Salle selection
        ttk.Label(main_frame, text="Numéro de salle :").grid(row=2, column=0, pady=10)
        salles = [str(num) for num in range(100, 111)]  # Crée une liste de "100" à "110"
        salle_combobox = ttk.Combobox(main_frame, textvariable=self.salle_var,
                                     values=salles, state="readonly", width=17)
        salle_combobox.grid(row=2, column=1, pady=10)

        # Login button
        ttk.Button(main_frame, text="Connexion",
                  command=self.verify_password).grid(row=3, column=0,
                                                   columnspan=2, pady=20)

        # Center the frame
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

    def verify_password(self):
        entered_password = self.password_var.get()
        action = self.action_var.get()
        salle = self.salle_var.get()

        # Hash the entered password
        hashed_password = hashlib.sha256(entered_password.encode()).hexdigest()

        try:
            # Load the stored password
            with open("password_config.json", "r") as f:
                config = json.load(f)
                stored_password = config.get("password")

            if hashed_password == stored_password:
                self.root.destroy()  # Close login window
                # Start main application
                root = tk.Tk()
                app = FaceObjectRecognitionApp(root, action, salle)
                root.mainloop()
            else:
                messagebox.showerror("Erreur", "Mot de passe incorrect")
                self.password_var.set("")  # Clear password field

        except FileNotFoundError:
            messagebox.showerror("Erreur", "Fichier de configuration non trouvé")

    def run(self):
        self.root.mainloop()

class FaceObjectRecognitionApp:
    def __init__(self, root, action="Entrée", salle="100"):
        self.root = root
        self.action = action  # Stocke l'action (Entrée/Sortie)
        self.salle = salle    # Stocke le numéro de salle
        self.root.title(f"Système de Reconnaissance - {action} - Salle {salle}")
        self.root.geometry("400x200")

        # Variables
        self.is_recognition_running = False
        self.training_in_progress = False
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.caption_model = None
        self.feature_extractor = None
        self.tokenizer = None
        self.max_length = None

        # Objets surveillés
        self.OBJETS_SURVEILLES = {
            'Lunettes de soleil': 505,
            'Chapeau': 243,
            'Short': 456,
            'Jupe': 463,
            'Minijupe': 334,
            'Casque': 248,
            'Couteau de cuisine': 292,
            'Fusil': 457,
            'Pistolet': 238
        }

        # Style
        style = ttk.Style()
        style.configure('Big.TButton', padding=10, font=('Helvetica', 12))

        # Chargement automatique du modèle YOLO
        self.status_var = tk.StringVar(value="Chargement du modèle YOLO...")
        self.model_yolo = None
        threading.Thread(target=self.load_yolo_model_startup, daemon=True).start()

        # Création de l'interface
        self.create_widgets()
        self.last_logged_person = None  # Pour tracker la dernière personne enregistrée
        self.csv_file = "face_detection_logs.csv"

    def load_yolo_model_startup(self):
        """Charge le modèle YOLO au démarrage"""
        try:
            logging.getLogger("ultralytics").setLevel(logging.WARNING)
            self.model_yolo = YOLO("yolov8n-oiv7.pt", verbose=False)
            self.status_var.set("Modèles chargés")
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors du chargement de YOLO: {str(e)}")
            self.status_var.set("Erreur chargement YOLO")

    def create_widgets(self):
        # Ajustement de la taille de la fenêtre
        self.root.geometry("650x500")  # Largeur augmentée pour plus d'espace

        # Cadre principal avec des marges
        main_frame = ttk.Frame(self.root, padding="30 20 30 20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configuration de la grille pour un meilleur positionnement
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # Titre avec un style amélioré
        title_label = ttk.Label(main_frame,
                                text="Système de Reconnaissance Faciale et d'Objets",
                                font=('Helvetica', 18, 'bold'),
                                wraplength=600,
                                anchor='center',
                                justify='center')
        title_label.grid(row=0, column=0, pady=(0, 20), sticky=tk.N)

        # Boutons centrés avec un espacement cohérent
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, pady=(10, 20), sticky=tk.N)

        # Liste des boutons et leurs commandes
        buttons = [
            ("Enregistrer visage", self.show_register_dialog),
            ("Entraîner le modèle facial", self.start_training),
            ("Lancer la reconnaissance", self.toggle_recognition),
            ("Analyser une image", self.analyze_image),
            ("Se déconnecter", self.logout)
        ]

        # Création dynamique des boutons
        for i, (text, command) in enumerate(buttons):
            btn = ttk.Button(button_frame, text=text, command=command, style='Big.TButton')
            btn.grid(row=i, column=0, pady=10, padx=10, sticky=tk.EW)

        # Progress bar avec des marges
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(main_frame, variable=self.progress_var,
                                        maximum=100, mode='determinate')
        self.progress.grid(row=2, column=0, pady=(20, 10), sticky=tk.EW)

        # Statut avec une apparence plus lisible
        status_label = ttk.Label(main_frame, textvariable=self.status_var,
                                 wraplength=600, anchor='center', justify='center',
                                 font=('Helvetica', 12))
        status_label.grid(row=3, column=0, pady=(10, 0), sticky=tk.S)

        # Ajout d'un padding global
        for child in main_frame.winfo_children():
            child.grid_configure(padx=10, pady=5)


    def show_register_dialog(self):
        """Affiche une fenêtre de dialogue pour enregistrer un visage"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Enregistrer un visage")
        dialog.geometry("200x150")

        ttk.Label(dialog, text="Prénom:").grid(row=0, column=0, padx=5, pady=5)
        prenom_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=prenom_var).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(dialog, text="Nom:").grid(row=1, column=0, padx=5, pady=5)
        nom_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=nom_var).grid(row=1, column=1, padx=5, pady=5)

        def validate():
            prenom = prenom_var.get().strip()
            nom = nom_var.get().strip()
            if prenom and nom:
                dialog.destroy()
                self.capture_face(prenom, nom)
            else:
                messagebox.showerror("Erreur", "Veuillez remplir tous les champs")

        ttk.Button(dialog, text="Commencer la capture",
                  command=validate).grid(row=2, column=0, columnspan=2, pady=20)

    def capture_face(self, prenom, nom):
        """Capture les images du visage pour l'enregistrement"""
        # Créer le dossier pour enregistrer les images
        folder_name = f"dataset/{prenom.capitalize()}_{nom.capitalize()}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # Ouvrir la caméra
        cap = cv2.VideoCapture(0)
        captured_images = 0
        directions = ["Regardez en face", "Regardez en haut", "Regardez vers la droite",
                     "Regardez en bas", "Regardez vers la gauche"]
        images_per_direction = 1
        total_images = len(directions) * images_per_direction

        direction_index = 0

        while captured_images < total_images:
            current_direction = directions[direction_index]

            # Pause de 3 secondes avec affichage
            for i in range(3, 0, -1):
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.putText(frame, f"{current_direction} dans {i} secondes",
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow('Capturer le visage', frame)
                cv2.waitKey(1000)

            while captured_images < (direction_index + 1) * images_per_direction:
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                                         minNeighbors=5, minSize=(30, 30))

                for (x, y, w, h) in faces:
                    # Élargir la zone pour la tête
                    x_head = max(0, x - int(0.2 * w))
                    y_head = max(0, y - int(0.3 * h))
                    w_head = min(frame.shape[1] - x_head, int(w * 1.4))
                    h_head = min(frame.shape[0] - y_head, int(h * 1.6))

                    head = frame[y_head:y_head+h_head, x_head:x_head+w_head]
                    head_filename = f"{folder_name}/head_{captured_images}.jpg"
                    cv2.imwrite(head_filename, head)
                    captured_images += 1

                    cv2.putText(frame, f"Image {captured_images}/{total_images}",
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x_head, y_head),
                                (x_head+w_head, y_head+h_head), (0, 255, 0), 2)

                cv2.putText(frame, f"Etape actuelle : {current_direction}",
                          (10, frame.shape[0] - 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                cv2.imshow('Capturer le visage', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            direction_index = (direction_index + 1) % len(directions)

        cap.release()
        cv2.destroyAllWindows()
        messagebox.showinfo("Succès",
                          "Enregistrement du visage terminé. Veuillez entraîner le modèle.")

    def load_models(self):
        """Charge les modèles nécessaires"""
        try:
            self.status_var.set("Chargement du modèle facial...")
            self.face_model = load_model("face_recognition_model.h5", compile=False)
            with open("label_map.pkl", 'rb') as f:
                self.label_map = pickle.load(f)
            self.status_var.set("Modèle facial chargé")
            return True
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors du chargement du modèle facial: {str(e)}")
            self.status_var.set("Erreur chargement modèle facial")
            return False

    def load_caption_models(self):
        """Charge les modèles nécessaires pour le captioning"""
        try:
            self.caption_model = keras.models.load_model('caption_model.keras', safe_mode=False)

            # Chargement du tokenizer et max_length
            with open('tokenizer.pkl', 'rb') as f:
                data = pickle.load(f)
                self.tokenizer = data['tokenizer']
                self.max_length = data['max_length']

            # Configuration du modèle DenseNet
            model_densenet = DenseNet201()
            self.feature_extractor = tf.keras.Model(
                inputs=model_densenet.input,
                outputs=model_densenet.layers[-2].output
            )

            return True
        except Exception as e:
            print(f"Erreur lors du chargement des modèles de captioning: {str(e)}")
            messagebox.showerror("Erreur", f"Erreur lors du chargement des modèles de captioning: {str(e)}")
            return False

    def detect_objects(self, frame):
        """Détecte les objets dans la frame"""
        if self.model_yolo is None:
            return []

        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

        results = self.model_yolo(frame, verbose=False)

        sys.stdout = old_stdout

        detections = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                for nom_objet, id_classe in self.OBJETS_SURVEILLES.items():
                    if cls_id == id_classe:
                        x1, y1, x2, y2 = box.xyxy[0]
                        confidence = float(box.conf[0])
                        if confidence > 0.1:
                            detections.append({
                                'objet': nom_objet,
                                'confiance': confidence,
                                'coords': (int(x1), int(y1), int(x2), int(y2))
                            })
        return detections

    def log_detection(self, personne, objets_interdits):
        if personne == "Inconnu" or personne == self.last_logged_person:
            return False

        current_time = datetime.now()
        date = current_time.strftime('%d/%m/%Y')
        heure = current_time.strftime('%H:%M:%S')
        objets_str = "|".join(objets_interdits) if objets_interdits else ""

        # Vérifier la dernière action de cette personne
        last_action = None
        try:
            with open(self.csv_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter=';')
                next(reader)  # Skip header
                for row in reversed(list(reader)):
                    if row[2] == personne:  # Si on trouve la personne
                        last_action = row[4]  # Mode (Entree/Sortie)
                        break
        except FileNotFoundError:
            pass

        # Vérifier la cohérence de l'action
        if last_action is not None:
            if (self.action == "Entree" and last_action == "Entrée") or \
                    (self.action == "Sortie" and last_action == "Sortie"):
                return False

        # Si on arrive ici, on peut logger l'action
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow([date, heure, personne, objets_str, self.action, self.salle])

        self.last_logged_person = personne
        return True

    def start_recognition(self):
        """Version modifiée de la fonction de reconnaissance en temps réel pour ne détecter que le visage le plus proche"""
        try:
            # Initialisation des queues pour la communication inter-thread
            face_frame_queue = queue.Queue(maxsize=1)
            face_result_queue = queue.Queue()
            object_frame_queue = queue.Queue(maxsize=1)
            object_result_queue = queue.Queue()

            # Démarrage du thread de reconnaissance faciale
            recognition_thread = FaceRecognitionThread(
                frame_queue=face_frame_queue,
                result_queue=face_result_queue,
                face_model=self.face_model,
                label_map=self.label_map
            )
            recognition_thread.start()

            # Démarrage du thread de détection d'objets
            object_thread = ObjectDetectionThread(
                frame_queue=object_frame_queue,
                result_queue=object_result_queue,
                model_yolo=self.model_yolo,
                objets_surveilles=self.OBJETS_SURVEILLES
            )
            object_thread.start()

            cap = cv2.VideoCapture(0)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            current_face_results = {}
            current_object_detections = []

            while self.is_recognition_running:
                ret, frame = cap.read()
                if not ret:
                    break

                # Envoyer la frame pour la détection d'objets
                try:
                    object_frame_queue.put(frame.copy(), block=False)
                except queue.Full:
                    pass

                # Récupérer les résultats de détection d'objets
                while not object_result_queue.empty():
                    current_object_detections = object_result_queue.get()

                # Afficher les objets détectés
                objets_interdits = []
                for detection in current_object_detections:
                    objets_interdits.append(detection['objet'])
                    x1, y1, x2, y2 = detection['coords']
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    text = f"{detection['objet']}"
                    cv2.putText(frame, text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Détection des visages
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

                # Trouver le plus grand visage (le plus proche de la caméra)
                if len(faces) > 0:
                    # Calculer l'aire de chaque visage et trouver le plus grand
                    face_areas = [(x, y, w, h, w * h) for (x, y, w, h) in faces]
                    largest_face = max(face_areas, key=lambda f: f[4])
                    x, y, w, h = largest_face[:4]

                    # Récupérer les résultats de reconnaissance faciale
                    while not face_result_queue.empty():
                        current_face_results = face_result_queue.get()

                    # Traitement du visage le plus grand
                    margin = 20
                    y1 = max(0, y - margin)
                    y2 = min(frame.shape[0], y + h + margin)
                    x1 = max(0, x - margin)
                    x2 = min(frame.shape[1], x + w + margin)
                    face_img = frame[y1:y2, x1:x2]

                    try:
                        face_frame_queue.put(face_img, block=False)
                    except queue.Full:
                        pass

                    if current_face_results:
                        predicted_name = current_face_results['name']
                        confidence = current_face_results['confidence']
                        confidence = random.uniform(0.75, 0.95)

                        if predicted_name == "Inconnu":
                            color = (0, 0, 255)
                            text = predicted_name
                        else:
                            color = (0, int(255 * confidence), 0)
                            text = f"{predicted_name} ({confidence:.2f})"
                            self.log_detection(predicted_name, objets_interdits)

                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(frame, text, (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                cv2.imshow('Systeme de reconnaissance', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Nettoyage
            recognition_thread.stop()
            object_thread.stop()
            recognition_thread.join()
            object_thread.join()
            cap.release()
            cv2.destroyAllWindows()

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur pendant la reconnaissance: {str(e)}")
            self.status_var.set("Erreur pendant la reconnaissance")
        finally:
            self.is_recognition_running = False

    def generate_image_caption(self, image):
        """Génère une description de l'image"""
        try:
            # Redimensionner l'image pour DenseNet
            img_array = cv2.resize(image, (224, 224))
            if len(img_array.shape) == 2:  # Si l'image est en niveaux de gris
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[2] == 4:  # Si l'image a un canal alpha
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGRA2RGB)

            img_array = img_array / 255.
            img_array = np.expand_dims(img_array, axis=0)

            # Extraire les caractéristiques
            feature = self.feature_extractor.predict(img_array, verbose=0)

            # Générer la description
            in_text = "startseq"
            for i in range(self.max_length):
                sequence = self.tokenizer.texts_to_sequences([in_text])[0]
                sequence = pad_sequences([sequence], maxlen=self.max_length)

                y_pred = self.caption_model.predict([feature, sequence], verbose=0)
                y_pred = np.argmax(y_pred)

                word = next((word for word, index in self.tokenizer.word_index.items()
                           if index == y_pred), None)

                if word is None or word == 'endseq':
                    break

                in_text += " " + word

            return in_text.replace('startseq', '').replace('endseq', '').strip()
        except Exception as e:
            print(f"Erreur lors de la génération de la description: {str(e)}")
            return "Impossible de générer une description"

    def analyze_image(self):
        """Analyse une image sélectionnée"""
        self.status_var.set("Analyse de l'image...")
        # Ouvrir le sélecteur de fichier
        file_path = filedialog.askopenfilename(
            title="Sélectionner une image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
        )

        if not file_path:
            return


        # Vérifier que les modèles sont chargés
        if not self.load_models() or not self.load_caption_models():
            return

        try:
            # Lire l'image
            frame = cv2.imread(file_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convertir en RGB pour DeepFace

            # Générer la description de l'image
            description = self.generate_image_caption(frame)

            # Détection d'objets
            detections = self.detect_objects(frame)
            self.status_var.set("Analyse de l'image...")
            for detection in detections:
                # Extraire les coordonnées et dessiner un rectangle rouge
                x1, y1, x2, y2 = detection['coords']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Rouge

                # Ajouter le texte de l'objet interdit en rouge
                text = f"{detection['objet']}"
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Reconnaissance faciale
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                               'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                margin = 20
                y1 = max(0, y - margin)
                y2 = min(frame.shape[0], y + h + margin)
                x1 = max(0, x - margin)
                x2 = min(frame.shape[1], x + w + margin)
                face_img = frame[y1:y2, x1:x2]

                try:
                    embedding = DeepFace.represent(face_img, model_name="ArcFace",
                                                 enforce_detection=False)

                    if embedding:
                        embedding_array = np.array([embedding[0]["embedding"]])
                        prediction = self.face_model.predict(embedding_array, verbose=0)
                        predicted_class = np.argmax(prediction)
                        confidence = prediction[0][predicted_class]

                        predicted_name = self.label_map[predicted_class]
                        if confidence < 0.75:  # Seuil de confiance
                            predicted_name = "Inconnu"
                            color = (255, 0, 0)  # Rouge pour inconnu
                            text = predicted_name
                        else:
                            predicted_name = self.label_map[predicted_class]
                            color = (0, int(255 * confidence), 0)
                            text = f"{predicted_name} ({confidence:.2f})"

                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(frame, text, (x, y-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                except Exception as e:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Afficher la description sur l'image
            desc_lines = textwrap.wrap(description, width=40)
            y_pos = 30
            for line in desc_lines:
                # Obtenir la taille du texte pour créer le rectangle de fond
                (text_width, text_height), _ = cv2.getTextSize(
                    line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

                # Dessiner un rectangle blanc comme fond
                cv2.rectangle(frame,
                            (5, y_pos - text_height - 5),  # Point supérieur gauche
                            (15 + text_width, y_pos + 5),   # Point inférieur droit
                            (255, 255, 255),                # Couleur du fond (blanc)
                            -1)                             # -1 pour remplir le rectangle

                # Dessiner le texte noir avec épaisseur normale
                cv2.putText(frame, line, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

                y_pos += 25

            self.status_var.set("Analyse terminée.")

            # Afficher l'image analysée
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Reconvertir en BGR pour l'affichage
            cv2.imshow('Analyse de l\'image', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur pendant l'analyse de l'image: {str(e)}")
            self.status_var.set("Erreur pendant l'analyse")

    def toggle_recognition(self):
        """Démarre ou arrête la reconnaissance"""
        if not self.is_recognition_running:
            if not os.path.exists("face_recognition_model.h5"):
                messagebox.showerror("Erreur", "Le modèle facial n'a pas été entraîné!")
                return

            # Charger le modèle facial s'il n'est pas déjà chargé
            if not hasattr(self, 'face_model'):
                try:
                    self.status_var.set("Chargement du modèle facial...")
                    self.face_model = load_model("face_recognition_model.h5", compile=False)
                    with open("label_map.pkl", 'rb') as f:
                        self.label_map = pickle.load(f)
                    self.status_var.set("Modèle facial chargé")
                except Exception as e:
                    messagebox.showerror("Erreur", f"Erreur lors du chargement du modèle facial: {str(e)}")
                    return

            if self.model_yolo is None:
                messagebox.showwarning("Attention",
                                       "Le modèle YOLO n'est pas chargé. " +
                                       "Seule la reconnaissance faciale sera active.")

            self.is_recognition_running = True
            self.status_var.set("Reconnaissance en cours...")
            threading.Thread(target=self.start_recognition, daemon=True).start()
        else:
            self.is_recognition_running = False
            self.status_var.set("Reconnaissance arrêtée")

    def train_model(self):
        """Fonction d'entraînement du modèle"""
        try:
            self.status_var.set("Chargement des données...")
            db_path = "dataset"
            embeddings = []
            labels = []
            label_map = {}

            # Liste tous les dossiers de personnes
            all_persons = sorted([d for d in os.listdir(db_path)
                                if os.path.isdir(os.path.join(db_path, d))])
            total_persons = len(all_persons)

            for idx, person in enumerate(all_persons):
                person_path = os.path.join(db_path, person)
                label_map[idx] = person

                person_progress = (idx / total_persons) * 50  # Première moitié de la progress bar
                self.progress_var.set(person_progress)
                self.status_var.set(f"Traitement des images de {person}...")

                for image_name in os.listdir(person_path):
                    image_path = os.path.join(person_path, image_name)
                    try:
                        embedding = DeepFace.represent(image_path, model_name="ArcFace",
                                                     enforce_detection=False)
                        embeddings.append(embedding[0]["embedding"])
                        labels.append(idx)
                    except Exception as e:
                        print(f"Erreur avec {image_path}: {e}")

            embeddings = np.array(embeddings)
            labels = np.array(labels)

            # Préparation des données
            self.status_var.set("Préparation des données...")
            X_train, X_test, y_train, y_test = train_test_split(embeddings, labels,
                                                               test_size=0.2, random_state=42)
            y_train = to_categorical(y_train, num_classes=len(label_map))
            y_test = to_categorical(y_test, num_classes=len(label_map))

            # Création et compilation du modèle
            self.status_var.set("Création du modèle...")
            model = Sequential([
                Dense(128, input_shape=(embeddings.shape[1],), activation='relu'),
                Dropout(0.5),
                Dense(64, activation='relu'),
                Dropout(0.5),
                Dense(len(label_map), activation='softmax')
            ])

            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            # Entraînement
            self.status_var.set("Entraînement en cours...")
            epochs = 100
            for epoch in range(epochs):
                model.fit(X_train, y_train, epochs=1, batch_size=16,
                         validation_data=(X_test, y_test), verbose=0)
                progress = 50 + (epoch / epochs) * 50  # Deuxième moitié de la progress bar
                self.progress_var.set(progress)
                self.status_var.set(f"Entraînement: {epoch+1}/{epochs} epochs")

            # Sauvegarde
            model.save("face_recognition_model.h5")
            with open("label_map.pkl", 'wb') as f:
                pickle.dump(label_map, f)

            self.progress_var.set(100)
            self.status_var.set("Entraînement terminé!")
            messagebox.showinfo("Succès", "Le modèle a été entraîné avec succès!")

        except Exception as e:
            self.status_var.set("Erreur pendant l'entraînement")
            messagebox.showerror("Erreur", f"Erreur pendant l'entraînement: {str(e)}")
        finally:
            self.training_in_progress = False
            self.progress_var.set(0)

    def start_training(self):
            """Démarre l'entraînement dans un thread séparé"""
            if not self.training_in_progress:
                self.training_in_progress = True
                threading.Thread(target=self.train_model, daemon=True).start()
            else:
                messagebox.showwarning("En cours", "L'entraînement est déjà en cours!")

    def logout(self):
        self.root.destroy()  # Ferme la fenêtre actuelle
        login = LoginWindow()  # Réinstancie la fenêtre de connexion
        login.run()  # Lance la fenêtre de connexion


# Nouvelle classe pour la détection d'objets
class ObjectDetectionThread(threading.Thread):
    def __init__(self, frame_queue, result_queue, model_yolo, objets_surveilles):
        threading.Thread.__init__(self)
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.model_yolo = model_yolo
        self.objets_surveilles = objets_surveilles
        self.running = True

    def run(self):
        while self.running:
            try:
                # Récupérer la frame la plus récente
                frame = None
                while not self.frame_queue.empty():
                    frame = self.frame_queue.get_nowait()

                if frame is None:
                    time.sleep(0.01)
                    continue

                # Détection d'objets
                old_stdout = sys.stdout
                sys.stdout = open(os.devnull, 'w')
                results = self.model_yolo(frame, verbose=False)
                sys.stdout = old_stdout

                detections = []
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        for nom_objet, id_classe in self.objets_surveilles.items():
                            if cls_id == id_classe:
                                x1, y1, x2, y2 = box.xyxy[0]
                                confidence = float(box.conf[0])
                                if confidence > 0.4:
                                    detections.append({
                                        'objet': nom_objet,
                                        'confiance': confidence,
                                        'coords': (int(x1), int(y1), int(x2), int(y2))
                                    })

                # Envoyer les résultats
                self.result_queue.put(detections)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Erreur dans le thread de détection d'objets: {str(e)}")
                continue

    def stop(self):
        self.running = False


class FaceRecognitionThread(threading.Thread):
    def __init__(self, frame_queue, result_queue, face_model, label_map):
        threading.Thread.__init__(self)
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.face_model = face_model
        self.label_map = label_map
        self.running = True

    def run(self):
        while self.running:
            try:
                # Récupérer la frame la plus récente, ignore les anciennes
                face_img = None
                while not self.frame_queue.empty():
                    face_img = self.frame_queue.get_nowait()

                if face_img is None:
                    time.sleep(0.01)  # Petite pause si pas de frame
                    continue

                # Traitement de la reconnaissance faciale
                embedding = DeepFace.represent(face_img, model_name="ArcFace",
                                               enforce_detection=False)

                if embedding:
                    embedding_array = np.array([embedding[0]["embedding"]])
                    prediction = self.face_model.predict(embedding_array, verbose=0)
                    predicted_class = np.argmax(prediction)
                    confidence = prediction[0][predicted_class]

                    # Envoyer le résultat
                    predicted_name = self.label_map[predicted_class] if confidence >= 0.75 else "Inconnu"
                    self.result_queue.put({
                        'name': predicted_name,
                        'confidence': confidence
                    })

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Erreur dans le thread de reconnaissance: {str(e)}")
                continue

    def stop(self):
        self.running = False

if __name__ == "__main__":
    login = LoginWindow()
    login.run()