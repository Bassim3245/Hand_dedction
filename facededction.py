import mediapipe as mp
import cv2
import os
import uuid

# Initialize MediaPipe tools
mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection

# Create output directory for saving images
output_dir = 'Output_Faces'
os.makedirs(output_dir, exist_ok=True)

def process_face_detection():
    """Function to capture video and perform face detection."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    with mp_face_detection.FaceDetection(min_detection_confidence=0.8) as face_detection:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read from the webcam.")
                break

            # Preprocess the frame
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)
            image.flags.writeable = False

            # Face detection
            results = face_detection.process(image)

            # Postprocess the frame
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw face detection annotations if detected
            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(image, detection)

                # Save the image
                filename = os.path.join(output_dir, f"{uuid.uuid1()}.jpg")
                cv2.imwrite(filename, image)

            # Display the processed frame
            cv2.imshow('Face Detection', image)

            # Exit on 'q' key press
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# if __name__ == "__main__":
#     process_face_detection()
