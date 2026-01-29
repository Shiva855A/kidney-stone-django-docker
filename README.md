🩺 Kidney Stone Detection Using CNN
📌 Project Overview

Kidney stone disease is a common and painful medical condition that requires early and accurate diagnosis. This project presents a deep learning–based web application for detecting kidney stones from medical images using a Convolutional Neural Network (CNN). The system is built using Django for the web interface and TensorFlow/Keras for image classification.

The application allows users to upload kidney scan images and receive predictions indicating the presence or absence of kidney stones, along with additional analysis such as estimated stone size, severity, and treatment recommendations.

🎯 Objectives

To automate kidney stone detection using medical images

To apply CNN-based image classification for healthcare diagnosis

To integrate machine learning models into a web application

To provide user-friendly prediction and analysis results

🧠 Technologies Used
Programming & Frameworks

Python 3

Django (Web Framework)

Machine Learning

TensorFlow

Keras

Convolutional Neural Network (CNN)

Image Processing

OpenCV

NumPy

Pillow

Frontend

HTML

CSS

Bootstrap

Database

SQLite

⚙️ System Architecture

User uploads a kidney scan image through the web interface

Image is preprocessed and resized

CNN model analyzes the image

Prediction result is generated

Additional image analysis determines:

Stone size

Severity level

Kidney region

Results are displayed to the user

🧪 Machine Learning Model

Model Type: Convolutional Neural Network (CNN)

Input Size: 128 × 128 RGB images

Output: Binary classification (Stone / No Stone)

Loss Function: Binary Cross-Entropy

Optimizer: Adam

Activation Functions: ReLU, Sigmoid

The trained model is saved as:

kidney_stone_model.keras

📊 Features

User registration and login

Image upload and prediction

Kidney stone detection with confidence score

Stone size estimation

Severity classification (Mild / Moderate / Severe)

Treatment recommendation

Secure and structured Django application

📂 Project Structure
KSD/
│── KSD/                 # Django project settings
│── Users/               # User management app
│── Admins/              # Admin-related features
│── templates/           # HTML templates
│── static/              # CSS, JS, images
│── media/               # Uploaded images
│── temp_images/         # Temporary prediction images
│── manage.py
│── requirements.txt
│── kidney_stone_model.keras

🔍 Dataset

Medical kidney scan images

Binary classes:

Kidney Stone

Normal

Images are resized and normalized before training

📈 Results

High accuracy in detecting kidney stones from images

Efficient and fast prediction through web interface

Helpful diagnostic insights for medical analysis

🔮 Future Enhancements

Support for multi-class stone classification

Integration with real-time hospital systems

Advanced visualization of detected stone regions

Deployment on cloud healthcare platforms

Mobile application integration

🧠 Academic Relevance

This project demonstrates the practical application of:

Deep Learning in healthcare

CNN-based image classification

Django–ML integration

Medical image analysis

👨‍💻 Author

Shiva Kumar
Python Developer | Machine Learning Enthusiast

📄 License

This project is developed for academic and educational purposes only.
