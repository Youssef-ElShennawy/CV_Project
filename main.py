from utils.ui_setup import create_empty_canvas
from utils.cam_launch import show_camera_feed
import cv2

def main():
    canvas = create_empty_canvas()

    # Show the initial canvas window
    cv2.imshow("AI Virtual Painter - Canvas", canvas)

    # Start the camera feed (hand tracking + drawing)
    show_camera_feed(canvas)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
