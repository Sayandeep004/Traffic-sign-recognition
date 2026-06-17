# Traffic-sign-recognition
This is my initiative to create a CNN model capable of predicting 43 different types of traffic sign

This project implements a real-time traffic sign recognition system using a Convolutional Neural Network (CNN). The model is trained on the German Traffic Sign Recognition Benchmark (GTSRB) dataset and is capable of recognizing 43 different types of traffic signs.
Features

  Image Upload: Upload an image for traffic sign recognition.
  Live Video Detection: Detect traffic signs in real-time using your webcam.
  User Interface: A simple and intuitive GUI built using Tkinter.
  Preprocessing: Images are resized and normalized before prediction.
  Real-time Display: The recognized sign is displayed on both uploaded images and live video feeds.

Technologies Used

  TensorFlow/Keras: For building and training the CNN model.
  OpenCV: For capturing and processing video frames.
  Tkinter: For creating the user interface.
  NumPy & PIL: For image manipulation and handling.

How to Run


Clone the repository:
bash
git clone https://github.com/yourusername/traffic-sign-recognition.git
cd traffic-sign-recognition

Install dependencies:
bash
pip install -r requirements.txt

Run the application:
bash
python traffic_sign_recognition.py

To be noted:-
Before running the script the dataset location needs to be changed according to your dataset location on your device.

Future Improvements

 1. Improving model accuracy in low-light and occluded environments.
 2. Adding support for more traffic sign categories.
 3. Enhancing the live video detection for better performance on low-end devices.
