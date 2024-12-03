# Import necessary libraries
import os
import cv2
import numpy as np
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import mediapipe as mp
# Initialize Mediapipe Hand module for hand detection and tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# Define folder path for dataset
EXTRACTED_FOLDER = r"D:\New folder\pythonHand\leapgestrecog" # this folder or path for dataset can ypu change base path for pc
print(f"Dataset Directory: {EXTRACTED_FOLDER}")
# Dynamically get gesture names from folder names in the dataset directory
gesture_names = sorted([d for d in os.listdir(EXTRACTED_FOLDER) if os.path.isdir(os.path.join(EXTRACTED_FOLDER, d))])
print(f"Gestures found: {gesture_names}")
# Function to preprocess the dataset
def preprocess_dataset(dataset_dir, target_size=(64, 64)):
    """
    Load images, resize, normalize, and prepare labels.
    Args:
        dataset_dir: Directory containing gesture folders.
        target_size: Target size to resize images.
    Returns:
        Preprocessed image data and corresponding labels.
    """
    data = []
    labels = []
    for gesture_id, gesture_folder in enumerate(gesture_names):
        gesture_folder_path = os.path.join(dataset_dir, gesture_folder)
        if os.path.isdir(gesture_folder_path):
            print(f"Processing folder: {gesture_folder_path}")
            for img_file in os.listdir(gesture_folder_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(gesture_folder_path, img_file)
                    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if image is not None:
                        try:
                            resized_image = cv2.resize(image, target_size)
                            data.append(resized_image)
                            labels.append(gesture_id)
                        except Exception as e:
                            print(f"Error resizing image {img_path}: {e}")
                    else:
                        print(f"Failed to load image: {img_path}")
    # Convert to numpy arrays
    data = np.array(data)
    labels = np.array(labels)
    # Normalize pixel values
    data = data / 255.0
    # Reshape data to match CNN input requirements
    data = data.reshape(-1, *target_size, 1)
    return data, labels
# Load and preprocess the dataset
X, y = preprocess_dataset(EXTRACTED_FOLDER)
print(f"Dataset loaded: {len(X)} images, {len(y)} labels.")
# Split dataset into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
print(f"Train: {len(X_train)} images, Validation: {len(X_val)} images, Test: {len(X_test)} images.")
# Function to build the CNN model
def build_cnn_model(input_shape=(64, 64, 1), num_classes=len(gesture_names)):
    """
    Build a CNN model for gesture recognition.
    Args:
        input_shape: Shape of input images.
        num_classes: Number of gesture categories.

    Returns:
        Compiled CNN model.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),  # Increased filter size for better feature extraction
        layers.Flatten(),
        layers.Dense(128, activation='relu'),  # Increased neurons for better learning capacity
        layers.Dropout(0.5),  # Dropout to prevent overfitting
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
# Build and summarize the model
model = build_cnn_model()
model.summary()
# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=32,
                    validation_data=(X_val, y_val))  # Increased epochs for better learning
# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
# Save the trained model
model_path = 'hand_gesture_model.h5'
model.save(model_path)
print(f"Model saved at {model_path}.")
# Function to recognize gestures using the trained model
def recognize_gesture_with_cnn(image, model):
    """
    Recognize gestures from an input image using the CNN model.
    Args:
        image: Preprocessed image of the hand.
        model: Trained CNN model.

    Returns:
        Predicted gesture ID and confidence score.
    """
    input_image = cv2.resize(image, (64, 64)) / 255.0
    input_image = input_image.reshape(-1, 64, 64, 1)
    prediction = model.predict(input_image)
    gesture_id = np.argmax(prediction)
    return gesture_id, prediction[0][gesture_id]
# Hand tracking and gesture recognition
def process_hand_tracking_with_cnn(model):
    """
    Perform hand tracking and gesture recognition using Mediapipe and CNN.
    Args:
        model: Trained CNN model.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read from the webcam.")
                break

            # Flip frame for natural interaction
            frame = cv2.flip(frame, 1)

            # Convert frame to RGB for Mediapipe processing
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks on the frame
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
                    )

                    # Generate a blank canvas for landmarks
                    hand_image = np.zeros((64, 64), dtype=np.uint8)
                    for point in hand_landmarks.landmark:
                        cx, cy = int(point.x * 64), int(point.y * 64)
                        cv2.circle(hand_image, (cx, cy), 2, 255, -1)

                    try:
                        gesture_id, confidence = recognize_gesture_with_cnn(hand_image, model)
                        gesture_name = gesture_names[gesture_id]
                        cv2.putText(frame, f"Gesture: {gesture_name} ({confidence * 100:.2f}%)",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    except Exception as e:
                        print(f"Error recognizing gesture: {e}")
            cv2.imshow('Hand Tracking with CNN', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
# Start the hand tracking process
process_hand_tracking_with_cnn(model)
