# 🌿 Plant Disease Detector (Custom CNN)

An end-to-end Machine Learning pipeline to identify plant diseases from leaf images. Built from scratch using **PyTorch** and deployed as an interactive web application via **Streamlit**.

## 🎯 Project Overview
This project leverages Deep Learning to help in early plant disease detection. The model is trained on a dataset of over 43,000 images and can classify leaves into **38 distinct categories** (healthy and diseased).

## 🚀 Performance
* **Model:** Custom Convolutional Neural Network (CNN)
* **Accuracy:** **84.33%** on Validation Data

## 🧠 Model Architecture
The custom CNN architecture includes:
* 2 Convolutional Layers (Conv2d) with ReLU activation
* Max Pooling layers for spatial downsampling
* Fully Connected (Linear) layer for the final 38-class prediction

## 🛠️ Tech Stack
* **Deep Learning Framework:** PyTorch, Torchvision
* **Web Deployment:** Streamlit
* **Image Processing:** Pillow (PIL)

## 💻 How to Run This Project Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/shankardaya4170-ux/Plant-Disease-Detector.git](https://github.com/shankardaya4170-ux/Plant-Disease-Detector.git)
   cd Plant-Disease-Detector
