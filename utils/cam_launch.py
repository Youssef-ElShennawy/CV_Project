import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from utils.ui_setup import draw_toolbar, TOOLBAR_BUTTONS

# --- Global State for Drawing ---

# Queues to store points for different colors
# We use a list of deques for each color to handle multiple, disconnected strokes
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]
all_points = [bpoints, gpoints, rpoints, ypoints]

# Indices to track the current stroke for each color
blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0
all_indices = [blue_index, green_index, red_index, yellow_index]

# BGR color values for drawing
colors_bgr = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)] # Blue, Green, Red, Yellow
colorIndex = 0 # Current active color index (0=Blue, 1=Green, 2=Red, 3=Yellow)

# Dimensions
WIDTH = 640
HEIGHT = 480

# --- MediaPipe Setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def reset_drawing_state(canvas):
    """Resets all drawing deques and indices, and clears the canvas drawing area."""
    global bpoints, gpoints, rpoints, ypoints
    global blue_index, green_index, red_index, yellow_index

    bpoints = [deque(maxlen=1024)]
    gpoints = [deque(maxlen=1024)]
    rpoints = [deque(maxlen=1024)]
    ypoints = [deque(maxlen=1024)]
    
    # Update the global list reference
    all_points[0] = bpoints
    all_points[1] = gpoints
    all_points[2] = rpoints
    all_points[3] = ypoints

    blue_index = 0
    green_index = 0
    red_index = 0
    yellow_index = 0
    all_indices[0] = blue_index
    all_indices[1] = green_index
    all_indices[2] = red_index
    all_indices[3] = yellow_index

    # Clear the canvas below the toolbar (y > 67)
    canvas[67:, :, :] = 255

def append_new_stroke():
    """Starts a new disconnected stroke for all colors."""
    global blue_index, green_index, red_index, yellow_index
    
    bpoints.append(deque(maxlen=1024))
    blue_index += 1
    gpoints.append(deque(maxlen=1024))
    green_index += 1
    rpoints.append(deque(maxlen=1024))
    red_index += 1
    ypoints.append(deque(maxlen=1024))
    yellow_index += 1

    all_indices[0] = blue_index
    all_indices[1] = green_index
    all_indices[2] = red_index
    all_indices[3] = yellow_index


def show_camera_feed(canvas):
    """
    Main loop for camera feed, hand tracking, and drawing logic.
    """
    global colorIndex

    cap = cv2.VideoCapture(0)
    # Set to a fixed size for consistent landmark scaling
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    
    if not cap.isOpened():
        print("Camera not detected.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for mirror view
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Draw the UI on the live camera frame
        draw_toolbar(frame)

        # Process hand landmarks
        result = hands.process(framergb)

        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    # Scale landmarks to frame size (WIDTH x HEIGHT)
                    lmx = int(lm.x * WIDTH)
                    lmy = int(lm.y * HEIGHT)
                    landmarks.append((lmx, lmy))

                # Drawing landmarks on frames
                mp_draw.draw_landmarks(frame, handslms, mp_hands.HAND_CONNECTIONS)
            
            # Index finger tip (Landmark 8)
            index_finger = landmarks[8]
            # Thumb tip (Landmark 4)
            thumb = landmarks[4]
            center = index_finger

            # Draw a circle on the index finger for visual feedback
            cv2.circle(frame, center, 5, (0, 255, 0), -1)

            # --- 1. Eraser/New Stroke Detection (Thumb close to Index finger) ---
            # Check the vertical distance between index finger and thumb
            if abs(thumb[1] - center[1]) < 30: # If the hand is closed for "erasing" or moving stroke
                append_new_stroke()

            # --- 2. Toolbar Click Detection (Pointing to the toolbar area) ---
            elif center[1] <= 65:
                for i, (x1, y1, x2, y2, _, _, _, _, idx) in enumerate(TOOLBAR_BUTTONS):
                    if x1 <= center[0] <= x2:
                        if idx == -1: # Clear Button
                            reset_drawing_state(canvas)
                        else: # Color Button
                            colorIndex = idx
                        break
            
            # --- 3. Drawing Mode (Pointing to the canvas area) ---
            else:
                current_points = all_points[colorIndex]
                current_index = all_indices[colorIndex]
                current_points[current_index].appendleft(center)

        # --- No Hand Detected / Append New Stroke ---
        else:
            # If no hand is detected, we append a new stroke to disconnect lines
            append_new_stroke()

        # --- Draw all stored lines on the canvas and frame ---
        for i in range(len(all_points)): # Loop through colors
            points = all_points[i]
            color = colors_bgr[i]
            for j in range(len(points)): # Loop through strokes of that color
                for k in range(1, len(points[j])): # Loop through points in the stroke
                    p1 = points[j][k - 1]
                    p2 = points[j][k]
                    if p1 is None or p2 is None:
                        continue
                    # Draw on the live frame
                    cv2.line(frame, p1, p2, color, 4)
                    # Draw on the persistent canvas
                    cv2.line(canvas, p1, p2, color, 4)


        # Display the result windows
        cv2.imshow("AI Virtual Painter - Camera", frame)
        cv2.imshow("AI Virtual Painter - Canvas", canvas)

        # Press 'q' to exit
        if cv2.waitKey(1) == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
