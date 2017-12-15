import numpy as np
import time
import cv2


cap = cv2.VideoCapture('../Footage/TestSeq2.mp4')
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=25, detectShadows=False)

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

start = time.time()

while True:

    ret, frame = cap.read()
    if not ret:
        break

    im = frame.copy()

    frame = fgbg.apply(frame)

    _, contours, hierarchy = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('frame', im)
    cv2.imshow('BGSub', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


end = time.time()
print("[INFO] it took %s seconds." % (end - start))
print("[INFO] clip has %s frames" % frameCount)
print("[INFO] that makes %s fps" % (frameCount / (end - start)))

cap.release()
cv2.destroyAllWindows()
