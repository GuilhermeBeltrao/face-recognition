import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from keras_facenet import FaceNet

def load_images_from_folder(folder):
    images = []
    labels = []
    label_map = {}
    label_id = 0

    for person in os.listdir(folder):
        person_folder = os.path.join(folder, person)
        if os.path.isdir(person_folder):
            label_map[label_id] = person
            for filename in os.listdir(person_folder):
                img_path = os.path.join(person_folder, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (160, 160))
                    images.append(img)
                    labels.append(label_id)
            label_id += 1

    return np.array(images), np.array(labels), label_map

def get_embedding(facenet_model, face_pixels):
    face_pixels_rgb = cv2.cvtColor(face_pixels, cv2.COLOR_BGR2RGB)
    embedding = facenet_model.embeddings([face_pixels_rgb])[0]
    return embedding

def plot_confusion_matrix(cm, title, label_map):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=label_map.values(), yticklabels=label_map.values())
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.show()

print("Loading FaceNet model...")
facenet = FaceNet()

print("Loading images...")
images, labels, label_map = load_images_from_folder("dataset_faces")
print(f"Loaded {len(images)} images from {len(label_map)} different people")

print("Generating face embeddings...")
embeddings = np.array([get_embedding(facenet, img) for img in images])

print("Splitting dataset for training and testing...")

X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=15, stratify=labels)

print("Training KNN classifier...")
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

print("Training SVM classifier...")
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

acc_knn = accuracy_score(y_test, y_pred_knn)
acc_svm = accuracy_score(y_test, y_pred_svm)

print("\n===== Individual Model Performance =====")
print(f"KNN Accuracy: {acc_knn:.4f}")
print(f"SVM Accuracy: {acc_svm:.4f}")

better_model = "KNN" if acc_knn > acc_svm else "SVM"
print(f"The better performing model is: {better_model}")

agreement_count = 0
disagreement_count = 0

print("\n===== Model Comparison =====")
print(f"KNN Accuracy: {acc_knn:.4f}")
print(f"SVM Accuracy: {acc_svm:.4f}")

cm_knn = confusion_matrix(y_test, y_pred_knn)
cm_svm = confusion_matrix(y_test, y_pred_svm)

plot_confusion_matrix(cm_knn, "Confusion Matrix - KNN", label_map)
plot_confusion_matrix(cm_svm, "Confusion Matrix - SVM", label_map)


print("\n===== Misclassification Analysis =====")
for i in range(len(y_test)):
    if y_pred_knn[i] != y_test[i] or y_pred_svm[i] != y_test[i]:
        true_person = label_map[y_test[i]]
        knn_pred = label_map[y_pred_knn[i]]
        svm_pred = label_map[y_pred_svm[i]]
        
        print(f"Image {i}: True person: {true_person}")
        print(f"  - KNN predicted: {knn_pred}")
        print(f"  - SVM predicted: {svm_pred}")
        print("---")