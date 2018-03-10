from imutils import resize
import numpy as np
import time
import cv2

cap = cv2.VideoCapture('../../Footage/TestSeq1.mp4')
# vid = cap.open("/media/victor/57a90e07-058d-429d-a357-e755d0820324/Footage/TestSeq1.mp4")
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

ret, firstFrame = cap.read()
firstFrame = resize(cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY), width=500)
cv2.imshow("test", firstFrame)
frameNumber = 0

start = time.time()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    (ho, wo) = frame.shape[:2]

    frame = resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # compute the absolute difference between the current frame and
    # first frame
    frameDelta = cv2.absdiff(firstFrame, gray)
    cv2.imshow('frame diff', frameDelta)

    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    _, cnts, val = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # loop over the contours
    for c in cnts[2]:
        # if the contour is too small, ignore it
        # if cv2.contourArea(c) < 1:
        #     continue

        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.1)

    # Finally update the framenumber, so the csv writer knows at which frame we are
    frameNumber += 1

end = time.time()
print("[INFO] it took %s seconds." % (end - start))
print("[INFO] clip has %s frames" % frameCount)
print("[INFO] that makes %s fps" % (frameCount / (end - start)))

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()