import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from PIL import Image
import cv2
from keras_facenet import FaceNet 
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

print("Loading FaceNet model...")
facenet = FaceNet()

def get_embedding(facenet_model, face_pixels):
    """
    Get the embedding from the FaceNet model
    The keras-facenet package expects RGB images but OpenCV loads as BGR
    """
    face_pixels_rgb = cv2.cvtColor(face_pixels, cv2.COLOR_BGR2RGB)
    embedding = facenet_model.embeddings([face_pixels_rgb])[0]
    return embedding

print("Loading images...")
images, labels, label_map = load_images_from_folder("dataset_faces")
print(f"Loaded {len(images)} images from {len(label_map)} different people")

print("Generating face embeddings...")
embeddings = []
for i, img in enumerate(images):
    if i % 10 == 0:
        print(f"Processing image {i+1}/{len(images)}")
    embedding = get_embedding(facenet, img)
    embeddings.append(embedding)
embeddings = np.array(embeddings)
print(f"Generated {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")

print("Splitting dataset for training and testing...")
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=5*3, stratify=labels, random_state=42)
print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

print("Training KNN classifier...")
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
acc_knn = accuracy_score(y_test, y_pred_knn)
cm_knn = confusion_matrix(y_test, y_pred_knn)
print("Accuracy KNN:", acc_knn)
print("Confusion Matrix KNN:\n", cm_knn)

print("Training SVM classifier...")
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
cm_svm = confusion_matrix(y_test, y_pred_svm)
print("Accuracy SVM:", acc_svm)
print("Confusion Matrix SVM:\n", cm_svm)


# pra deixar bonitinho
print("\n===== Model Comparison =====")
if acc_knn > acc_svm:
    print(f"KNN performs better with {acc_knn:.4f} accuracy (SVM: {acc_svm:.4f})")
    best_model = "KNN"
elif acc_svm > acc_knn:
    print(f"SVM performs better with {acc_svm:.4f} accuracy (KNN: {acc_knn:.4f})")
    best_model = "SVM"
else:
    print(f"Both models perform equally with {acc_knn:.4f} accuracy")
    best_model = "Either KNN or SVM"

print(f"Recommendation: Use {best_model} for this face recognition task")