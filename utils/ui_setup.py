import cv2
import numpy as np

def draw_toolbar(frame):
    """Draws the toolbar and quit text on the camera feed."""
    cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), 2)
    cv2.rectangle(frame, (160, 1), (255, 65), (255, 0, 0), 2)
    cv2.rectangle(frame, (275, 1), (370, 65), (0, 255, 0), 2)
    cv2.rectangle(frame, (390, 1), (485, 65), (0, 0, 255), 2)
    cv2.rectangle(frame, (505, 1), (600, 65), (0, 255, 255), 2)

    cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    cv2.putText(frame, "Press any key to quit", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)


def create_empty_canvas(width=960, height=720):
    """Creates an empty white canvas."""
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
    cv2.putText(canvas, "Press any key to quit", (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)
    return canvas
