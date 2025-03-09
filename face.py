import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from PIL import Image
import cv2

def load_images_from_folder(folder):
    images = []
    labels = []
    label_map = {}
    label_id = 0
    
    # Handling the selfies and resizing them because facenet only works with 160x160 images
    for person in os.listdir(folder):
        person_folder = os.path.join(folder, person)
        if os.path.isdir(person_folder):
            label_map[label_id] = person
            for filename in os.listdir(person_folder):
                img_path = os.path.join(person_folder, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (160, 160))  # it might have took me a while to figure this out
                    images.append(img)
                    labels.append(label_id)
            label_id += 1
    
    return np.array(images), np.array(labels), label_map


# facenet_model = models.load_model("facenet_keras.h5")
facenet_model = tf.keras.models.load_model("facenet_keras.h5")


def get_embedding(model, face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    face_pixels = np.expand_dims(face_pixels, axis=0)
    return model.predict(face_pixels)[0]

images, labels, label_map = load_images_from_folder("dataset_faces")
embeddings = np.array([get_embedding(facenet_model, img) for img in images])


X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=5*3, stratify=labels, random_state=42)



# Classificador kNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
acc_knn = accuracy_score(y_test, y_pred_knn)
cm_knn = confusion_matrix(y_test, y_pred_knn)
print("Accuracy KNN:", acc_knn)
print("Matriz de Confusão KNN:\n", cm_knn)

# Classificador SVM
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
cm_svm = confusion_matrix(y_test, y_pred_svm)
print("Accuracy SVM:", acc_svm)
print("Matriz de Confusão SVM:\n", cm_svm)

#TODO: maybe compare the models?
