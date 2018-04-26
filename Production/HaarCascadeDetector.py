import Utils
from imutils import resize
import numpy as np
import cv2


class HaarCascadeDetector:
    def __init__(self, winstride=(4, 4), padding=(8, 8), scale=1.05, classifierfile='haarcascade_upperbody.xml'):
        self.winstride = winstride
        self.padding = padding
        self.scale = scale
        self.haar = cv2.CascadeClassifier(classifierfile)

    def getName(self):
        return "Haar"

    def detect(self, frame, height, width):
        # (ho, wo) = frame.shape[:2]
        # originalFrame = frame

        frame = resize(frame, width=min(400, frame.shape[1]))
        (hr, wr) = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        widthScale = width / wr
        heightScale = height / hr

        rects = self.haar.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=7, minSize=(75, 75))

        # apply non-maxima suppression to the bounding boxes using a
        # fairly large overlap threshold to try to maintain overlapping
        # boxes that are still people

        rects = np.array([[x, y, x + w, y + h] * np.array([widthScale, heightScale, widthScale, heightScale])
                 for (x, y, w, h) in rects])
        # boxes = non_max_suppression(rects, probs=None, overlapThresh=0.65)
        boxes = Utils.non_max_suppression_fast(rects, overlapThresh=0.65)

        return boxes
