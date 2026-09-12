# Driver-Behavior-Detection
A deep learning based driver behavior classification system using  EfficientNetB0 and Streamlit.
## Project Overview

This project detects driver behaviors from images/videos using 
a transfer learning approach with EfficientNetB0.

The model classifies five driver behaviors:

- Safe Driving
- Talking Phone
- Texting Phone
- Turning
- Other Activities


## Dataset

Multi-Class Driver Behavior Image Dataset

Classes:

- talking_phone
- texting_phone
- other_activities
- safe_driving
- turning


## Model

Architecture:
- EfficientNetB0
- Transfer Learning
- Softmax classifier

Input size:
224 × 224 × 3

Framework:
- TensorFlow / Keras


## Training

The model was trained on Google Colab using GPU acceleration.

Training pipeline:

1. Dataset loading
2. Image preprocessing
3. Transfer learning with EfficientNetB0
4. Model evaluation
5. Model saving


## Deployment

The trained model was deployed using Streamlit.

Features:

✅ Upload driver videos  
✅ Frame-by-frame analysis  
✅ Behavior prediction  
✅ Confidence estimation  
✅ Generated analyzed video output  


## Technologies

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Streamlit


## Author

Sogol Tehranipoor
