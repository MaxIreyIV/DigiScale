import cv2
import numpy as np


def capture_image(device_index: int = 0,
                  width: int = 900,
                  height: int = 500) -> np.ndarray | None:
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return frame


if __name__ == "__main__":
    frame = capture_image()
    if frame is None:
        raise SystemExit("❌ Failed to capture image from camera.")

    cv2.imshow("Captured frame (press any key to close)", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()