import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
# Initialize MediaPipe tools
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
# Create output directory for saving images
output_dir = 'Output_Images'
os.makedirs(output_dir, exist_ok=True)
def is_hand_open(landmarks):
    """Check if the hand is open (all fingers extended)."""
    # Thumb, index, middle, ring, and pinky fingertips landmarks
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    # A hand is considered open if all the fingers' tips are extended
    is_open = thumb_tip.y < landmarks[0].y and \
              index_tip.y < landmarks[5].y and \
              middle_tip.y < landmarks[9].y and \
              ring_tip.y < landmarks[13].y and \
              pinky_tip.y < landmarks[17].y
    return is_open

def is_like_gesture(landmarks):
    """Check if the 'LIKE' gesture is made (thumb and index extended, others curled)."""
    # Thumb and index tip landmarks
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    # Thumb and index should be extended (above the base)
    is_thumb_extended = thumb_tip.y < landmarks[2].y  # Thumb is extended if its tip is higher than base

    # Middle, ring, and pinky should be curled (below the base)
    is_middle_curl = middle_tip.y > landmarks[9].y
    is_index_extended = index_tip.y > landmarks[5].y  # Index is extended if its tip is higher than base

    is_ring_curl = ring_tip.y > landmarks[13].y
    is_pinky_curl = pinky_tip.y > landmarks[17].y

    # Check if it's the "LIKE" gesture: Thumb and index extended, others curled
    if is_thumb_extended and is_index_extended and is_middle_curl and is_ring_curl and is_pinky_curl:
        return True
    return False
def process_hand_tracking():
    """Function to capture video and perform hand tracking."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return
    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read from the webcam.")
                break

            # Preprocess the frame
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)
            image.flags.writeable = False

            # Hand detection
            results = hands.process(image)
            # Postprocess the frame
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # Draw landmarks if detected
            if results.multi_hand_landmarks:
                for hand in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
                    )
                    # Check if the hand is open
                    if is_hand_open(hand.landmark):
                        # Display "HELLO" on the screen if the hand is open
                        cv2.putText(image, "HELLO", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

                    # Check if only the index finger is extended (LIKE gesture)
                    elif is_like_gesture(hand.landmark):
                        # Display "LIKE" on the screen if the index finger is extended
                        cv2.putText(image, "LIKE", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                # Save the image
                filename = os.path.join(output_dir, f"{uuid.uuid1()}.jpg")
                cv2.imwrite(filename, image)
            # Display the processed frame
            cv2.imshow('Hand Tracking', image)

            # Exit on 'q' key press
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    process_hand_tracking()
