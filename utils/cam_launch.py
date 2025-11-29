import cv2
import mediapipe as mp
from utils.ui_setup import draw_toolbar

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# For drawing lines
prev_point = None

# Current drawing color (default black)
current_color = (0, 0, 0)


def detect_toolbar_touch(x, y):
    """Returns the tool name based on finger position."""
    global current_color

    # Toolbar height range
    if 1 <= y <= 65:

        # CLEAR
        if 40 <= x <= 140:
            return "CLEAR"

        # BLUE
        if 160 <= x <= 255:
            current_color = (255, 0, 0)
            return "BLUE"

        # GREEN
        if 275 <= x <= 370:
            current_color = (0, 255, 0)
            return "GREEN"

        # RED
        if 390 <= x <= 485:
            current_color = (0, 0, 255)
            return "RED"

        # YELLOW
        if 505 <= x <= 600:
            current_color = (0, 255, 255)
            return "YELLOW"

    return None


def show_camera_feed(canvas, width=960, height=720):
    global prev_point, current_color

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return

    canvas_window = "AI Virtual Painter - Canvas"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (width, height))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:

                h, w, c = frame.shape
                lm_list = []
                for lm in handLms.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append((cx, cy))

                index_finger = lm_list[8]

                # Draw green circle on index finger
                cv2.circle(frame, index_finger, 10, (0, 255, 0), -1)

                # --- Toolbar interaction ---
                tool = detect_toolbar_touch(index_finger[0], index_finger[1])
                if tool == "CLEAR":
                    canvas[:] = 255
                    prev_point = None

                # Prevent drawing on toolbar area (top 70px)
                if index_finger[1] < 70:
                    prev_point = None
                else:
                    # Draw lines on canvas
                    if prev_point is None:
                        prev_point = index_finger
                    else:
                        cv2.line(canvas, prev_point, index_finger, current_color, 4)
                        prev_point = index_finger

                # Draw hand skeleton
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        else:
            prev_point = None

        # Draw toolbar on camera window
        draw_toolbar(frame)

        cv2.imshow("AI Virtual Painter - Camera", frame)
        cv2.imshow(canvas_window, canvas)

        # Press any key to quit
        if cv2.waitKey(1) != -1:
            break

    cap.release()
    cv2.destroyAllWindows()