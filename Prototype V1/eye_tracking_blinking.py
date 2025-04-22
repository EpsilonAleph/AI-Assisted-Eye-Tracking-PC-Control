import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

# Eye landmark indices
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# EAR Calculation Function
def calculate_ear(landmarks, eye_indices):
    p1 = np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y])
    p2 = np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y])
    p3 = np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y])
    p4 = np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])
    
    vertical_1 = np.linalg.norm(p2 - p4)
    vertical_2 = np.linalg.norm(p3 - p1)
    horizontal = np.linalg.norm(
        np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y]) -
        np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])
    )
    
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

# Blink detection parameters
BLINK_THRESHOLD = 0.48
# How long (in seconds) we wait after the last blink before evaluating the sequence
BLINK_EVALUATION_TIME = 0.4  

blink_counter = 0
last_blink_time = 0

left_clicks = 0
right_clicks = 0

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Calculate EAR for both eyes
            left_ear = calculate_ear(face_landmarks.landmark, LEFT_EYE_INDICES)
            right_ear = calculate_ear(face_landmarks.landmark, RIGHT_EYE_INDICES)
            
            # Draw eye landmarks for debugging
            for idx in LEFT_EYE_INDICES + RIGHT_EYE_INDICES:
                landmark = face_landmarks.landmark[idx]
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            
            # Detect a blink when both eyes are closed below the threshold
            if left_ear < BLINK_THRESHOLD and right_ear < BLINK_THRESHOLD:
                # Only count a new blink if sufficient time has passed since the previous one
                if time.time() - last_blink_time > 0.3:
                    blink_counter += 1
                    last_blink_time = time.time()
            
            # If enough time has passed since the last blink, evaluate the blink count
            if time.time() - last_blink_time > BLINK_EVALUATION_TIME and blink_counter > 0:
                if blink_counter == 2:
                    print("Left Click! 🔵")
                    pyautogui.click()
                    left_clicks += 1
                elif blink_counter == 3:
                    print("Right Click! 🔴")
                    pyautogui.rightClick()
                    right_clicks += 1
                # Reset blink counter after evaluation
                blink_counter = 0
            
            # Display debug info on the frame
            cv2.putText(frame, f"Left Eye EAR: {left_ear:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Right Eye EAR: {right_ear:.2f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Blinks: {blink_counter}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f"Left Clicks: {left_clicks}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Right Clicks: {right_clicks}", (10, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("Eye Tracking Mouse Control", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
