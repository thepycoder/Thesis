from imutils import resize
import numpy as np
import cv2


class HaarCascadeDetector:
    def __init__(self, name='bgsub'):
        self.detector = cv2.createBackgroundSubtractorKNN(detectShadows=True)
        self.name = name

    def getName(self):
        return self.name

    def detect(self, frame, height, width):
        # (ho, wo) = frame.shape[:2]
        # originalFrame = frame

        frame = resize(frame, width=min(400, frame.shape[1]))
        (hr, wr) = frame.shape[:2]

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        widthScale = width / wr
        heightScale = height / hr

        contourImage = self.detector.apply(frame)

        th, dst = cv2.threshold(contourImage, 127, 255, cv2.THRESH_BINARY)

        _, contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            box = np.array([[x, y, x + w, y + h] * np.array([widthScale, heightScale, widthScale, heightScale]))
            boxes.append(box)

        return boxes
