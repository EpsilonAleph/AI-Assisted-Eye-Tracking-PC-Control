import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Face Mesh with refined landmarks enabled
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Define eye landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Define iris landmark indices
LEFT_IRIS = [469, 470, 471, 472]
RIGHT_IRIS = [474, 475, 476, 477]

def get_eye_roi(frame, landmarks, eye_indices):
    #Extracts the eye region (ROI) using bounding box around given eye landmarks.
    h_frame, w_frame, _ = frame.shape
    points = np.array([[int(landmarks[i].x * w_frame), int(landmarks[i].y * h_frame)]
                        for i in eye_indices])
    x, y, w, h = cv2.boundingRect(points)
    return (x, y, w, h)

def get_horizontal_ratio(eye_points, frame, gray, landmarks):
    #Computes horizontal ratio using the white-pixel method.
    try:
        eye_region = np.array([[(landmarks[point].x * frame.shape[1],
                                  landmarks[point].y * frame.shape[0])
                                 for point in eye_points]], np.int32)
    except (IndexError, AttributeError):
        return None

    mask = np.zeros_like(gray)
    cv2.fillPoly(mask, eye_region, 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)
    
    x, y, w, h = cv2.boundingRect(eye_region)
    eye_roi = eye[y:y+h, x:x+w]
    
    _, thresh_eye = cv2.threshold(eye_roi, 70, 255, cv2.THRESH_BINARY)
    left_side = thresh_eye[:, :w//2]
    right_side = thresh_eye[:, w//2:]
    
    white_left = cv2.countNonZero(left_side)
    white_right = cv2.countNonZero(right_side)
    
    if white_right == 0:
        return 1
    return white_left / white_right

def get_iris_center(iris_indices, frame, landmarks):
    #Computes the iris center as the average of the iris landmark coordinates.
    h_frame, w_frame, _ = frame.shape
    try:
        points = [(landmarks[i].x * w_frame, landmarks[i].y * h_frame) for i in iris_indices]
    except (IndexError, AttributeError):
        return None
    return np.mean(points, axis=0)

def get_normalized_vertical_position(iris_center, eye_bbox):
    #Computes the normalized vertical position of the iris center within the eye ROI.
    x, y, w, h = eye_bbox
    if h == 0:
        return 0.5
    return (iris_center[1] - y) / h

def determine_horizontal_direction(avg_h_ratio):
    #Determines horizontal gaze direction using fixed thresholds.
    if avg_h_ratio < 0.50:
        return "Left"
    elif avg_h_ratio > 1.44:
        return "Right"
    return "Center"

def determine_vertical_direction(avg_norm_y, up_threshold=0.378, down_threshold=0.348, center_threshold_1=0.365, center_threshold_2=0.348):
    #Determines vertical gaze direction based on normalized vertical iris position.
    if avg_norm_y > up_threshold:
        return "Up"
    elif avg_norm_y < down_threshold:
        return "Down"
    elif center_threshold_2 < avg_norm_y < center_threshold_1:
        return "Center"

def combine_gaze(horz, vert):
    #The logic for combining horizontal and vertical gaze directions.
    if vert == "Center" and horz == "Center":
        return "Looking Center"
    elif vert == "Up" and horz == "Center":
        return "Looking Up"
    elif vert == "Down" and horz == "Center":
        return "Looking Down"
    elif vert == "Center" and horz == "Left":
        return "Looking Left"
    elif vert == "Center" and horz == "Right":
        return "Looking Right"
    elif vert == "Up" and horz == "Left":
        return "Looking Up-Left"
    elif vert == "Up" and horz == "Right":
        return "Looking Up-Right"
    elif vert == "Down" and horz == "Left":
        return "Looking Down-Left"
    elif vert == "Down" and horz == "Right":
        return "Looking Down-Right"
    return "Looking Center"  # Fallback case


# Main loop
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    result = face_mesh.process(rgb_frame)
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            
            # Horizontal gaze detection
            left_h_ratio = get_horizontal_ratio(LEFT_EYE, frame, gray, landmarks)
            right_h_ratio = get_horizontal_ratio(RIGHT_EYE, frame, gray, landmarks)
            if left_h_ratio is None or right_h_ratio is None:
                continue
            avg_h_ratio = (left_h_ratio + right_h_ratio) / 2
            horz_dir = determine_horizontal_direction(avg_h_ratio)
            
            # Vertical gaze detection using iris centers
            bbox_left = get_eye_roi(frame, landmarks, LEFT_EYE)
            bbox_right = get_eye_roi(frame, landmarks, RIGHT_EYE)

            left_iris_center = get_iris_center(LEFT_IRIS, frame, landmarks)
            right_iris_center = get_iris_center(RIGHT_IRIS, frame, landmarks)
            if left_iris_center is None or right_iris_center is None:
                continue

            norm_y_left = get_normalized_vertical_position(left_iris_center, bbox_left)
            norm_y_right = get_normalized_vertical_position(right_iris_center, bbox_right)
            avg_norm_y = (norm_y_left + norm_y_right) / 2
            vert_dir = determine_vertical_direction(avg_norm_y)
            
            # Combine gaze directions
            gaze = combine_gaze(horz_dir, vert_dir)
            
            # Display results on screen
            cv2.putText(frame, f"H Ratio: {avg_h_ratio:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(frame, f"V Pos: {avg_norm_y:.2f}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(frame, f"Gaze: {gaze}", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            print(f"H: {avg_h_ratio:.2f}, NormV: {avg_norm_y:.2f} -> {gaze}")
    
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
