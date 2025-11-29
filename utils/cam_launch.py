import cv2
import mediapipe as mp
from utils.ui_setup import draw_toolbar

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# For drawing continuous lines
prev_point = None


def show_camera_feed(canvas, width=960, height=720):
    global prev_point

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not detected.")
        return

    canvas_window = "AI Virtual Painter - Canvas"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (width, height))

        # Convert to RGB for Mediapipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # Hand detected
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:

                # Store all hand landmark points
                h, w, c = frame.shape
                lm_list = []
                for lm in handLms.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append((cx, cy))

                # Tip of the index finger → landmark 8
                index_finger = lm_list[8]

                # Draw green circle on index finger
                cv2.circle(frame, index_finger, 10, (0, 255, 0), -1)

                # Draw lines on the canvas (basic drawing)
                if prev_point is None:
                    prev_point = index_finger
                else:
                    cv2.line(canvas, prev_point, index_finger, (0, 0, 0), 4)
                    prev_point = index_finger

                # draw full hand connections
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        else:
            # Reset the stroke when hand disappears
            prev_point = None

        draw_toolbar(frame)   # toolbar visible but inactive (Week 3)

        # Display camera + canvas
        cv2.imshow("AI Virtual Painter - Camera", frame)
        cv2.imshow(canvas_window, canvas)

        if cv2.waitKey(1) != -1:
            break

    cap.release()
    cv2.destroyAllWindows()
