# 🕵️‍♂️ Deepfake Detector

An AI-powered Deepfake Detection system built with PyTorch, ResNet50, and deployed using Streamlit. This project identifies whether an image is **Real** or **Fake** with high accuracy.

---

## 📚 Project Overview

### ✨ Features
- Binary Classification using ResNet50.
- Real-time image classification with a confidence score.
- Training on downsampled datasets for efficiency.
- Custom Streamlit Web App for easy image uploads and predictions.
- Model checkpointing with early stopping to prevent overfitting.

### 🧠 Model
- **Architecture:** Pre-trained ResNet50 model.
- **Fine-Tuned Layer:** Fully connected (`fc`) layer modified for binary classification.
- **Loss Function:** `BCEWithLogitsLoss` for binary classification.

---

## 📂 Project Structure
![image](https://github.com/user-attachments/assets/f65065d9-f4ca-4688-920e-3da02afa5adc)
## 🛠️ Setup Instructions

### 1. 🚀 Cloning the repo
git clone https://github.com/aadilshaikh123/Deepfake-Detector.git
cd Deepfake-Detector
### Create virtual environment
python -m venv venv
### Activate it
# On Windows
venv\Scripts\activate
# On Mac/Linux
source venv/bin/activate
### 📚 Install Dependencies
pip install -r requirements.txt
### 🧑‍💻 Run the Streamlit App
streamlit run app.py
### 📊 Dataset
The dataset used for training can be found here: https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images
### 🔎 Usage
Run app.py and upload an image to classify it.
The model returns whether the image is Real or Fake along with the confidence score.
### 🧠 Training the Model
If you want to train the model from scratch:
python train.py
### 📡 Upload Model to GitHub Using Git LFS
If you modify and retrain the model:
### Make sure Git LFS is installed
git lfs install
### Track the model
git lfs track "model/*.pth"
### Add, commit, and push
git add .
git commit -m "Update model"
git push origin main

