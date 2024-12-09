import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import text

# Initialize MediaPipe Hands

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Initialize MediaPipe Drawing Module for drawing landmarks on the hands
mp_drawing = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the fram horizontally for more natural view
    frame = cv2.flip(frame,1)

    # Convert the BGR fram to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    # Process the frame and get the results
    results = hands.process(rgb_frame)

    # If hand landmarks are detected, draw them
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame with hand landmarks
    cv2.imshow("Hand Tracking", frame)

    # Exit the loop when 'q' if pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object and close any OpenCV windows
cap.release()
cv2.destroyAllWindows