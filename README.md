Dynamic Sign Language Recognition
A system for recognizing dynamic sign language gestures from video inputs using a BiLSTM model, with features extracted via MediaPipe and deployed on AWS using a Flask API.
Overview
This project implements a Dynamic Sign Language Recognition system that processes video inputs to predict sign language gestures. It uses a Bidirectional LSTM (BiLSTM) model trained on 2,319 video samples, with MediaPipe for feature extraction. The system is deployed on AWS, where a Flask API handles video uploads and returns predictions.
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

Installation

Clone the Repository:
git clone https://github.com/HyprerNova/Dynamic-Sign-Language-Recognition.git
cd Dynamic-Sign-Language-Recognition


Set Up Virtual Environment:
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt


Download Pre-trained Model:

Download the BiLSTM model from [link to model, e.g., S3 or Google Drive]
Place it in the models/ directory



Usage

Run the Flask API:
python app.py

API will be hosted at http://localhost:5000.

Upload a Video:Use cURL or Postman to send a video:
curl -X POST -F "video=@path/to/video.mp4" http://localhost:5000/predict


Response:
{
  "prediction": "gesture_name",
  "confidence": 0.95
}



API Endpoints

POST /predict
Input: Video file (multipart/form-data, key: video)
Output: JSON with predicted gesture and confidence
Example:curl -X POST -F "video=@sample_video.mp4" http://localhost:5000/predict





Model Details

Architecture: BiLSTM
Input: MediaPipe-extracted keypoints from video frames
Training: 2,319 video samples
Framework: TensorFlow/Keras
Performance: [Add metrics, e.g., accuracy, if available]

AWS Deployment

Services: [e.g., EC2 for Flask, S3 for storage]
Setup: See aws_deployment.md for detailed instructions
Access: API hosted at [your AWS endpoint, if public]

Project Structure
Dynamic-Sign-Language-Recognition/
├── app.py                    # Flask API
├── requirements.txt          # Dependencies
├── models/                   # Pre-trained BiLSTM model
├── src/                      # Source code
│   ├── feature_extraction.py # MediaPipe processing
│   ├── model.py             # BiLSTM model
│   └── predict.py           # Prediction logic
├── aws_deployment.md         # AWS setup guide
├── README.md                 # This file
├── LICENSE                   # License file
└── .gitignore                # Ignored files

Contributing

Fork the repository
Create a branch: git checkout -b feature-name
Commit changes: git commit -m "Add feature"
Push: git push origin feature-name
Open a Pull Request

License
MIT License
Contact

GitHub: HyprerNova
Email: [your.email@example.com]
