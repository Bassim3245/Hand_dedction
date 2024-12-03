# Hand Gesture Recognition System

This project implements a real-time hand gesture recognition system using a Convolutional Neural Network (CNN) and Mediapipe for hand tracking.

## Features
- Preprocesses a dataset of hand gesture images.
- Trains a CNN to classify hand gestures.
- Recognizes hand gestures in real time using webcam input.
- Displays gesture predictions with confidence percentages.

## Installation

--- 

### Improvements
1. **Error Handling**: Add more robust error handling for file I/O and webcam access.
2. **Model Optimization**: Experiment with hyperparameters like filter sizes, dropout rates, and learning rates.
3. **Augmentation**: Use data augmentation to improve model generalization.
4. **GPU Support**: Leverage GPU for faster training using TensorFlow-GPU.

Let me know if you need further assistance! 🚀

# Hand Gesture Recognition System

This project demonstrates a hand gesture recognition system using a Convolutional Neural Network (CNN) and Mediapipe's Hand module for real-time hand tracking and gesture classification.

## Features
- Real-time hand detection and tracking using Mediapipe.
- Gesture recognition with a trained CNN model.
- Dynamic dataset preparation and preprocessing.
- High accuracy on custom datasets with normalized input images.

## Code Explanation
The provided code demonstrates how to create a hand gesture recognition system. Below is a breakdown of the key components:

### 1. Import Libraries
- Essential libraries like TensorFlow, OpenCV, NumPy, Mediapipe, and Scikit-learn are used.
- Mediapipe is utilized for hand tracking, and TensorFlow is used for building and training the CNN model.

### 2. Dataset Preparation
- The dataset directory is specified (`EXTRACTED_FOLDER`).
- Gesture names are dynamically extracted from folder names in the dataset directory.
- **`preprocess_dataset` Function**:
  - Reads images from gesture folders.
  - Converts images to grayscale.
  - Resizes images to 64x64 pixels.
  - Normalizes pixel values to the range [0, 1].
  - Prepares labels corresponding to the gesture IDs.
- The dataset is split into training, validation, and test sets using `train_test_split`.

### 3. Building the CNN Model
- The **`build_cnn_model` Function**:
  - Contains three convolutional layers for feature extraction.
  - Includes max pooling layers for dimensionality reduction.
  - Uses a dropout layer to prevent overfitting.
  - Adds a fully connected layer for classification.
  - Outputs predictions with a softmax activation for multi-class output.
- The model is compiled with the Adam optimizer and sparse categorical cross-entropy loss.

### 4. Model Training
- The model is trained on the training dataset (`X_train`, `y_train`).
- Validation is performed using the validation dataset (`X_val`, `y_val`).
- Training is conducted over 14 epochs with a batch size of 32.

### 5. Testing the Model
- The model's accuracy is evaluated on the test dataset (`X_test`, `y_test`).

### 6. Gesture Recognition
- The **`recognize_gesture_with_cnn` Function**:
  - Predicts gestures for input images using the trained CNN model.
  - Resizes and normalizes input images before inference.

### 7. Real-Time Hand Tracking and Recognition
- Mediapipe's Hands module detects and tracks hands in a video feed.
- Hand landmarks are drawn on the video frame.
- For each detected hand:
  - A blank canvas maps the landmarks.
  - The CNN predicts the gesture.
  - The predicted gesture and confidence score are displayed on the video feed.

### 8. Execution
- The program captures video input from the webcam and processes each frame in real time.
- Press 'Q' to exit the webcam view.

## Requirements
- TensorFlow
- Mediapipe
- OpenCV
- NumPy
- Scikit-learn

## Installation
1. Clone the repository.
2. Install dependencies using:
   ```bash
   pip install -r requirements.txt

### Requirements
- Python 3.7+
- Required libraries:
  - TensorFlow :https://www.tensorflow.org/ 
  - Mediapipe
  - OpenCV:https://opencv.org/
  - NumPy:https://numpy.org/
  - Scikit-learn:https://scikit-learn.org/stable/

### Install Dependencies
```bash
pip install tensorflow mediapipe opencv-python numpy scikit-learn


