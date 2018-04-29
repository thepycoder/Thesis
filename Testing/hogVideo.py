from imutils.object_detection import non_max_suppression
from imutils import resize
import numpy as np
import time
import cv2
import csv


print("[INFO] loading model...")
hog = cv2.HOGDescriptor()
# hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# svm = cv2.ml.SVM_load("ALL.dat")
# sv = svm.getSupportVectors()

# detector = np.vstack(sv[0])
detector = np.loadtxt("../SVMs/50-50.dat")
hog.setSVMDetector(detector)

cap = cv2.VideoCapture()
vid = cap.open("/home/victor/Projects/Footage/Clips1/00:00:30.442.mp4")
# vid = cap.open("/media/victor/57a90e07-058d-429d-a357-e755d0820324/Footage/TestSeq1.mp4")
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

## Create empty dict to dump detections into so we can evaluate the performance

f = open("Results/resultsHog.csv", "w")
reader = csv.writer(f)
frameNumber = 0

start = time.time()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    (ho, wo) = frame.shape[:2]
    originalFrame = frame

    frame = resize(frame, width=min(400, frame.shape[1]))
    (hr, wr) = frame.shape[:2]

    widthScale = wo / wr
    heightScale = ho / hr

    (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] * np.array([widthScale, heightScale, widthScale, heightScale]) for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        # add the detection to the csv file
        # reader.writerow([frameNumber, xA, yA, xB, yB])
        cv2.rectangle(originalFrame, (xA, yA), (xB, yB), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame', originalFrame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Finally update the framenumber, so the csv writer knows at which frame we are
    frameNumber += 1

end = time.time()
print("[INFO] it took %s seconds." % (end - start))
print("[INFO] clip has %s frames" % frameCount)
print("[INFO] that makes %s fps" % (frameCount / (end - start)))

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()