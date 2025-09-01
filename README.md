Dynamic Kannada Sign Language Recognition

The Sign Language Recognition (SLR) of Dynamic Kannada Sign Language (KSL) words is done using three deep learning models, namely LSTM, BiLSTM and Transformers. All 3 models have been trained on a dataset of 3960 videos that consists of various signers performing common signs used in day to day communication.

The Signs have been divided into 4 sub-categories : Fruits, Months, Time and Weekdays. There are multiple classes under each folder in the sub-category. In total there are 33 classes and each class consists of 120 videos, thus totaling to 3960 videos.

This repo contains the following:
-> Code for feature extraction
-> npy files of the ectracted features
-> Model architecture and it's code 
-> Code for model demo using API calls
