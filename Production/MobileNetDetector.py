import numpy as np
import cv2


class MobileNetDetector:
    def __init__(self,
                 prototxt = "../Models/MobileNetSSD_deploy.prototxt",
                 caffemodel = "../Models/MobileNetSSD_deploy.caffemodel",
                 conf = 0.4):
        self.net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        self.conf = conf

    def getName(self):
        return "MobileNet"

    def detect(self, frame, height, width):
        # Create a blob from the source frame by resizing to the required 300x300 size
        # blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 2.0 / 255.0, mean=127)

        # Feed blob to the net and perform a forward pass
        self.net.setInput(blob)
        detections = self.net.forward()

        # List of bounding boxes
        boxes = []

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]
            classindex = int(detections[0, 0, i, 1])

            # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
            # Also suppress any output that is not a person
            if confidence > self.conf and classindex == 15:
                # Compute the (x, y)-coordinates of the bounding box for the object
                # box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                box = detections[0, 0, i, 3:7] * [width, height, width, height]
                # boxes.append(box.astype("int"))
                boxes.append(list(map(int, box)))

        return boxes
