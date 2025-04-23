import cv2
import numpy as np
import mediapipe as mp
import time
import winsound

# KEYBOARD SETTING/VARIABLES:
keyboard = np.zeros((300, 800, 3), np.uint8)
# Column indices remain the same
first_col_index   = [0,10,20,30,40,50]
second_col_index  = [1,11,21,31,41,51]
third_col_index   = [2,12,22,32,42,52]
fourth_col_index  = [3,13,23,33,43,53]
fifth_col_index   = [4,14,24,34,44,54]
sixth_col_index   = [5,15,25,35,45,55]
seventh_col_index = [6,16,26,36,46,56]
eighth_col_index  = [7,17,27,37,47,57]
ninth_col_index   = [8,18,28,38,48,58]
tenth_col_index   = [9,19,29,39,49,59]

# Key mapping remains the same
key_set = {
    0:"1",  1:"2",  2:"3",  3:"4",  4:"5",  5:"6",  6:"7",  7:"8",  8:"9",  9:"0",
   10:"q", 11:"w", 12:"e", 13:"r", 14:"t", 15:"y", 16:"u", 17:"i", 18:"o", 19:"p",
   20:"a", 21:"s", 22:"d", 23:"f", 24:"g", 25:"h", 26:"j", 27:"k", 28:"l", 29:";",
   30:"z", 31:"x", 32:"c", 33:"v", 34:"b", 35:"n", 36:"m", 37:"<", 38:">", 39:"?",
   40:"+", 41:"-", 42:",", 43:".", 44:"/", 45:"*", 46:"@", 47:".", 48:"!", 49:" ",
   50:"->",51:"->",52:"->",53:"->",54:"->",55:"->",56:"->",57:"->",58:"->",59:"->",
}

# Modified state variables
frame_count = 0
current_column = 0  # Always start from first column
current_row = 0
selection_mode = "COLUMN"  # Can be "COLUMN" or "ROW"
blink_counter = 0
last_blink_time = 0
selected_column = None
type_text = ""
white_board = np.ones((100,800,3), np.uint8)

# MediaPipe Face Mesh initialization
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye indices for EAR calculation
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

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
    return (vertical_1 + vertical_2) / (2.0 * horizontal)

# Blink detection parameters
BLINK_THRESHOLD = 0.45
FRAME_RATE = 30  # Increased from 10 to 30 for slower cycling
HIGHLIGHT_COLOR = (0, 165, 255)  # Orange color for highlighting

# Video capture
cap = cv2.VideoCapture(0)

def draw_keyboard(letter_index, letter, highlight=False):
    x = (letter_index % 10) * 80
    y = (letter_index // 10) * 60
    font = cv2.FONT_HERSHEY_PLAIN
    letter_thickness = 2
    key_space = 3
    font_scale = 3
    height = 60
    width = 80
    
    if highlight:
        cv2.rectangle(keyboard, (x + key_space, y + key_space), 
                     (x + width - key_space, y + height - key_space), 
                     HIGHLIGHT_COLOR, -1)
    else:
        cv2.rectangle(keyboard, (x + key_space, y + key_space), 
                     (x + width - key_space, y + height - key_space), 
                     (0, 0, 245), key_space)
    
    # Draw letter
    letter_size = cv2.getTextSize(letter, font, font_scale, letter_thickness)[0]
    letter_x = int((width - letter_size[0]) / 2) + x
    letter_y = int((height + letter_size[1]) / 2) + y
    cv2.putText(keyboard, letter, (letter_x, letter_y), font, font_scale, 
                (255, 255, 255), letter_thickness)

while True:
    main_windows = np.zeros((780, 1000, 3), np.uint8)
    keyboard.fill(0)  # Clear keyboard
    
    # Increment frame counter
    frame_count = (frame_count + 1) % FRAME_RATE
    
    # Update current column/row position
    if frame_count == 0:
        if selection_mode == "COLUMN":
            current_column = (current_column + 1) % 10
        elif selection_mode == "ROW":
            current_row = (current_row + 1) % 6

    # Get current column indices
    col_indices = [first_col_index, second_col_index, third_col_index, 
                  fourth_col_index, fifth_col_index, sixth_col_index, 
                  seventh_col_index, eighth_col_index, ninth_col_index, 
                  tenth_col_index][current_column]

    # Draw keyboard with highlights
    for i in range(60):
        if selection_mode == "COLUMN":
            highlight = i in col_indices
        else:  # ROW mode
            if selected_column is not None:
                highlight = i == selected_column[current_row]
            else:
                highlight = False
        draw_keyboard(i, key_set[i], highlight)

    # Process video frame
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_ear = calculate_ear(face_landmarks.landmark, LEFT_EYE_INDICES)
            right_ear = calculate_ear(face_landmarks.landmark, RIGHT_EYE_INDICES)

            # Detect blink
            if left_ear < BLINK_THRESHOLD and right_ear < BLINK_THRESHOLD:
                if time.time() - last_blink_time > 0.3:  # Minimum time between blinks
                    blink_counter += 1
                    last_blink_time = time.time()
                    
                    # Handle selection based on mode
                    if selection_mode == "COLUMN":
                        selected_column = col_indices
                        selection_mode = "ROW"
                        current_row = 0
                        frame_count = 0  # Reset frame count for row cycling
                        winsound.Beep(350, 65)  # Higher pitch for column selection
                    
                    elif selection_mode == "ROW":
                        # Select the letter
                        selected_letter = key_set[selected_column[current_row]]
                        type_text += selected_letter
                        selection_mode = "COLUMN"  # Reset to column selection
                        selected_column = None
                        current_column = 0  # Reset to first column
                        frame_count = 0  # Reset frame count for column cycling
                        winsound.Beep(280, 65)  # Lower pitch for letter selection
                        
                        # Update display text
                        white_board.fill(0)
                        cv2.putText(white_board, type_text, (10, 50),
                                  cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 3)

    # Display status
    status_text = f"Mode: {selection_mode}"
    cv2.putText(main_windows, status_text, (10, 30), 
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    # Update display
    main_windows[0:300, 300:700] = cv2.resize(frame, (400, 300))
    main_windows[350:650, 100:900] = keyboard
    main_windows[670:770, 100:900] = white_board
    
    cv2.imshow("Main_Windows", main_windows)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()