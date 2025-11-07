import cv2
from utils.ui_setup import draw_toolbar

def show_camera_feed(width=960, height=720):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (width, height))
        draw_toolbar(frame)
        cv2.imshow("AI Virtual Painter - Camera", frame)

        if cv2.waitKey(1) != -1:
            break

    cap.release()
    cv2.destroyAllWindows()
