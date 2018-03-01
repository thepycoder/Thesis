from imutils.object_detection import non_max_suppression
from imutils import resize
import numpy as np
import utils
import time
import cv2
import csv

class PeopleCounter:

    def __init__(self):
        prototxt = "MobileNetSSD_deploy.prototxt"
        model = "MobileNetSSD_deploy.caffemodel"
        self.net = cv2.dnn.readNetFromCaffe(prototxt, model)
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.detections = []

    def countInVideo(self, videoPath, showVideo = True):
        cap = cv2.VideoCapture()
        vid = cap.open(videoPath)
        framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        framenumber = 0

        start = time.time()

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            if not ret:
                break

            boxes = self.hogDetector(frame, framenumber)

            # draw the final bounding boxes
            for (xA, yA, xB, yB) in boxes:
                # add the detection to the csv file
                self.detections.append([framenumber, xA, yA, xB, yB])
                cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

            # Display the resulting frame
            if showVideo:
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Finally update the frame number, so the csv writer knows at which frame we are
            framenumber += 1

        end = time.time()

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

    def hogDetector(self, frame):
        (ho, wo) = frame.shape[:2]
        originalFrame = frame

        frame = resize(frame, width=min(400, frame.shape[1]))
        (hr, wr) = frame.shape[:2]

        widthScale = wo / wr
        heightScale = ho / hr

        (rects, weights) = self.hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

        # apply non-maxima suppression to the bounding boxes using a
        # fairly large overlap threshold to try to maintain overlapping
        # boxes that are still people
        rects = np.array(
            [[x, y, x + w, y + h] * np.array([widthScale, heightScale, widthScale, heightScale]) for (x, y, w, h) in
             rects])
        boxes = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        return boxes

    def mobileNetDetector(self, frame):
        # Define the confidence threshold parameter
        conf = 0.4

        # Get the bounding box list ready
        boxes = []

        # Get the frame dimensions for later on
        (h, w) = frame.shape[:2]

        # Create a blob from the source frame by resizing to the required 300x300 size
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

        self.net.setInput(blob)
        detections = self.net.forward()

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
            if confidence > conf:
                # Compute the (x, y)-coordinates of the bounding box for the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                boxes.append(box)


if __name__ == '__main__':
    counter = PeopleCounter()
    vid = "/media/victor/57a90e07-058d-429d-a357-e755d0820324/Footage/TestSeq1.mp4"
    counter.countInVideo(vid)
