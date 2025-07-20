Dynamic Sign Language Recognition
A system for recognizing dynamic sign language gestures from video inputs using a BiLSTM model, with features extracted via MediaPipe and deployed on AWS using a Flask API.
Overview. This project implements a Dynamic Sign Language Recognition system that processes video inputs to predict sign language gestures. It uses a Bidirectional LSTM (BiLSTM) model trained on 2,319 video samples, with MediaPipe for feature extraction. The system is deployed on AWS, where a Flask API handles video uploads and returns predictions.

<img width="930" height="863" alt="aws" src="https://github.com/user-attachments/assets/0c433932-8fe1-4840-829d-22ebb64d977b" />

Features

Real-time gesture recognition from video inputs
Feature extraction using MediaPipe for hand and body landmarks
BiLSTM model for accurate sequence modeling
Flask API for video uploads and predictions
Scalable deployment on AWS

Technologies

Python: Core programming language
MediaPipe: Feature extraction from videos
BiLSTM: Implemented in TensorFlow/Keras
Flask: API for video uploads and predictions
AWS: Hosting (e.g., EC2, S3)
OpenCV: Video processing
Git: Version control

Dataset

Size: 2,319 video samples of sign language gestures
Preprocessing: Features extracted using MediaPipe (hand/body landmarks)
Source: [Specify source, e.g., custom dataset or public dataset like WLASL]
Access: [Provide link or instructions to access dataset, if available]


Model Details

Architecture: BiLSTM
Input: MediaPipe-extracted keypoints from video frames
Training: 2,319 video samples
Framework: TensorFlow/Keras
Performance: 99% training accuracy and 91% testing accuracy.

<img width="1248" height="684" alt="image" src="https://github.com/user-attachments/assets/12cee6a3-b349-444b-9a9a-03267f1598f7" />

<img width="1138" height="1030" alt="image" src="https://github.com/user-attachments/assets/6832457e-9cb2-49fc-857b-11a6cb5e8b7c" />
