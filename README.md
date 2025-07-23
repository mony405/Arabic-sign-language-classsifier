# 🤟 Arabic Sign Language Image Classifier

This project builds a machine learning model to classify Arabic Sign Language images. It uses image preprocessing, feature extraction, and training with multiple classification algorithms to select the most accurate model, which is then deployed using a simple GUI built with Gradio.

---

## 📌 Table of Contents

* [Project Overview](#project-overview)
* [Dataset](#dataset)
* [Preprocessing](#preprocessing)
* [Model Training](#model-training)
* [Evaluation](#evaluation)
* [Deployment](#deployment)
* [How to Run](#how-to-run)
* [Requirements](#requirements)
* [Results](#results)
* [License](#license)
* [Author](#author)

---

## 📁 Project Overview

The goal of this project is to classify hand gesture images representing Arabic Sign Language characters. We built and compared four different machine learning models using scikit-learn and selected the best-performing model for deployment.

---

## 🖼️ Dataset

The dataset contains labeled images of Arabic Sign Language characters. Each image represents a single sign corresponding to an Arabic letter or word.

---

## 🔧 Preprocessing

* Loaded and resized the images.
* Applied normalization and flattening for feature extraction.
* Encoded the labels using `LabelEncoder`.
* Visualized class distribution using `matplotlib`.

---

## 🤖 Model Training

We tested and compared four classification algorithms:

* **Support Vector Machine (SVM)**
* **K-Nearest Neighbors (KNN)**
* **Random Forest Classifier**
* **XGBoost Classifier**

For each model:

* Applied `GridSearchCV` for hyperparameter tuning.
* Used consistent data splitting and cross-validation.
* Selected the best model based on test set performance.

✅ **Best Model: K-Nearest Neighbors (KNN)**

---

## 📊 Evaluation

The final KNN model was evaluated using:

* **Classification Report** (Precision, Recall, F1-Score)
* **Confusion Matrix**
* **Learning Curve**

All preprocessing tools and the trained model were saved using `joblib`.

---

## 🚀 Deployment

The selected model was deployed using **Gradio**, offering a user-friendly interface for uploading images and viewing predictions.

---

## ▶️ How to Run

```bash
# Clone the repository
git clone https://github.com/yourusername/arabic-sign-language-classifier.git
cd arabic-sign-language-classifier

# Install dependencies
pip install -r requirements.txt

# Run the notebook or launch the Gradio app
python app.py
```

---

## 📦 Requirements

* Python 3.8+
* scikit-learn
* xgboost
* matplotlib
* numpy
* pandas
* gradio
* joblib


## ✅ Results

* **Best Model:** K-Nearest Neighbors (KNN)
* **Accuracy:** 97%
* **F1-Score:** Averaged around 0.97
* **Confusion Matrix:** Clearly shows strong prediction performance across all classes with minimal misclassification.
* **ROC Curve:** AUC close to 1.0 for all classes, indicating excellent performance.
---
## Social media post
* [Linkedin Post](https://www.linkedin.com/feed/update/urn:li:activity:7353690326449950722/)
---
## YouTube Video 
* [Linkedin Post](https://youtu.be/1zNSCmUJYdg?si=Lcji73Fzi8-fPKZ6)
---
## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 👩‍💻 Author

* Menna Samir -> structure of the notebook ,data preprocessing ,model training ,best model evaluation, best model deployment(gui)
* Mohamed Tamer ->uploading the data ,data preprocessing, presentation
