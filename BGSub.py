from imutils.object_detection import non_max_suppression
import numpy as np
import time
import cv2


cap = cv2.VideoCapture()
cap = cv2.VideoCapture('../Footage/TestSeq2.mp4')
# vid = cap.open("/media/victor/57a90e07-058d-429d-a357-e755d0820324/Footage/TestSeq2.mp4")

frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=50, detectShadows=False)
sizeThreshold = 10000

#fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

start = time.time()

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    contourImage = frame.copy()

    contourImage = fgbg.apply(contourImage)

    # cv2.threshold(contourImage, 127, 255, cv2.THRESH_BINARY)

    _, contours, hierarchy = cv2.findContours(contourImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # rects = np.array([[cv2.boundingRect(cnt)] for cnt in contours])
    # test = np.array([[x, y, x + w, y + h] for [[x, y, w, h]] in rects])
    # pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # for [[x, y, w, h]] in rects:
    #     if w*h > sizeThreshold:
    #         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > sizeThreshold:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('frame', contourImage)
    cv2.imshow('BGSub', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(0.1)


end = time.time()
print("[INFO] it took %s seconds." % (end - start))
print("[INFO] clip has %s frames" % frameCount)
print("[INFO] that makes %s fps" % (frameCount / (end - start)))

cap.release()
cv2.destroyAllWindows()
