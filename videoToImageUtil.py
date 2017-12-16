################
#
#   videoToImageUtil.py is a small script to convert any video file to a sequence of still images
#   This is mostly used to annotate the video frame by frame in a normal annotation tool and to
#   generate the right annotations for evaluating the different algorithms
#
################

import cv2


cap = cv2.VideoCapture()
vid = cap.open("/media/victor/57a90e07-058d-429d-a357-e755d0820324/Footage/TestSeq1.mp4")

filename = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imwrite("/media/victor/57a90e07-058d-429d-a357-e755d0820324/Footage/TestSeq1/%s.jpg" % filename, frame)

    filename += 1