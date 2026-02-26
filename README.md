# Brain Tumor Detection using CNN

## Project Overview
This project implements a Convolutional Neural Network (CNN) model to classify MRI brain images into Tumor and No Tumor categories. 
The goal is to automate brain tumor detection using deep learning techniques.

## Dataset
- MRI Brain Tumor dataset
- Two classes: Tumor and No Tumor
- Images were resized and normalized before training
- The dataset is not included in this repository due to size limitations

## Model Architecture
The CNN model includes:
- Convolutional (Conv2D) layers
- MaxPooling layers
- Dropout layer for regularization
- Fully Connected (Dense) layers
- Softmax activation function

## Model Performance
- Training Accuracy: 93%
- Validation Accuracy: 87%

Note: The trained model file (.h5) is not included due to GitHub file size limitations.

## Tech Stack
- Python
- TensorFlow / Keras
- NumPy
- OpenCV

## How to Run the Project

1. Install required dependencies:
   pip install -r requirements.txt

2. Run the application:
   python app.py

## Author
Ashwin Deshpande
