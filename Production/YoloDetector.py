import numpy as np
import cv2

class YoloDetector:
    def __init__(self,
                 cfg = "../Models/yolov2-tiny.cfg",
                 weights = "../Models/yolov2-tiny.weights",
                 conf = 0.4):
        self.net = cv2.dnn.readNetFromDarknet(cfg, weights)
        self.conf = conf

    def getName(self):
        return "Yolo"

    def detect(self, frame, height, width):
        # Create a blob from the source frame by resizing to the required 300x300 size
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (416, 416)), 1.0 / 255.0)

        # Feed blob to the net and perform a forward pass
        self.net.setInput(blob)
        detections = self.net.forward()

        # List of bounding boxes
        boxes = []

        for i in range(detections.shape[0]):
            probability_index = 5
            probability_size = detections.shape[1] - probability_index

            objectClass = np.argmax(detections[i][probability_index:])
            confidence = detections[i][probability_index + objectClass]

            if confidence > 0.2 and objectClass == 0:
                x_center = detections[i][0] * width
                y_center = detections[i][1] * height
                width_det = detections[i][2] * width
                height_det = detections[i][3] * height

                # p1 = (int(x_center - width_det / 2), int(y_center - height_det / 2))
                # p2 = (int(x_center + width_det / 2), int(y_center + height_det / 2))

                # cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)

                boxes.append([int(x_center - width_det / 2), int(y_center - height_det / 2),
                              int(x_center + width_det / 2), int(y_center + height_det / 2)])

        return boxes