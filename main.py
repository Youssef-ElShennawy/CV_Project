from utils.ui_setup import create_empty_canvas
from utils.cam_launch import show_camera_feed
import cv2

def main():
    # Define consistent dimensions
    WIDTH = 640
    HEIGHT = 480
    
    # 1. Create the drawing canvas
    canvas = create_empty_canvas(WIDTH, HEIGHT)

    # Show the initial canvas window (must be shown before the loop starts)
    cv2.imshow("AI Virtual Painter - Canvas", canvas)

    # 2. Start the camera feed, hand tracking, and drawing logic
    show_camera_feed(canvas)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
