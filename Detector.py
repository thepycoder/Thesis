import numpy as np
import cv2


class MobileNetDetector:

    def __init__(self, model="MobileNetSSD_deploy.caffemodel", prototxt="MobileNetSSD_deploy.prototxt"):
        self.model = model
        self.prototxt = prototxt

        self.net = cv2.dnn.readNetFromCaffe(prototxt, model)

    def detect(self, image):
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)