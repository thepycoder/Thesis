from __future__ import print_function
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

cap = cv2.VideoCapture()
# vid = cap.open("../Footage/TestSeq1.mp4")
vid = cap.open("/media/victor/57a90e07-058d-429d-a357-e755d0820324/Footage/TestSeq1.mp4")
output = 'output.avi'
fps = 24

frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# use ffmpeg -i filename to get framerate

i = 0
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
writer = None
(h, w) = (None, None)
zeros = None

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break


    # check if the writer is None
    if writer is None:
        if i % 6 == 0:
            outputFrame = frame

    # store the image dimensions, initialzie the video writer,
    #  and construct the zeros array
    (h, w) = frame.shape[:2]
    writer = cv2.VideoWriter(output, fourcc, fps, (w, h), True)

    # write the output frame to file
    writer.write(outputFrame)

    i += 1


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()