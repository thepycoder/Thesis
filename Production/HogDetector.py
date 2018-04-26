import Utils
from imutils import resize
import numpy as np
import cv2


class HogDetector:
    def __init__(self, winstride=(4, 4), padding=(8, 8), scale=1.05):
        self.winstride = winstride
        self.padding = padding
        self.scale = scale
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        # self.hog.setSVMDetector(np.loadtxt("../SVMs/50-50.dat"))

    def getName(self):
        return "Hog"

    def detect(self, frame, height, width):
        # (ho, wo) = frame.shape[:2]
        # originalFrame = frame

        frame = resize(frame, width=min(400, frame.shape[1]))
        (hr, wr) = frame.shape[:2]

        widthScale = width / wr
        heightScale = height / hr

        (rects, weights) = self.hog.detectMultiScale(frame, winStride=self.winstride, padding=self.padding, scale=self.scale)

        # apply non-maxima suppression to the bounding boxes using a
        # fairly large overlap threshold to try to maintain overlapping
        # boxes that are still people
        rects = np.array([[x, y, x + w, y + h] * np.array([widthScale, heightScale, widthScale, heightScale])
                 for (x, y, w, h) in rects])
        # boxes = non_max_suppression(rects, probs=None, overlapThresh=0.65)
        boxes = Utils.non_max_suppression_fast(rects, overlapThresh=0.65)

        return boxes
