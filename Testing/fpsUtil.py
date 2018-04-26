################
#
#   fpsUtil.py is a quick tool to convert a high framerate video file to a very low framerate video
#   for use in testing counting algorithms when applied to very low framerates as might be found in
#   real life applications
#
################


import cv2

cap = cv2.VideoCapture()
vid = cap.open("/home/victor/Projects/Footage/Clips1/00:01:43.464.mp4")
# vid = cap.open("/media/victor/57a90e07-058d-429d-a357-e755d0820324/Footage/TestSeq1.mp4")
fps = 24

frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# use ffmpeg -i filename to get framerate

i = 0
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

# output filename, videowriter object, framse per second of out and dimensions of out
# out = cv2.VideoWriter('../Footage/TestSeq8.mp4', fourcc, 24.0, (1280, 720))
out = cv2.VideoWriter('/home/victor/Projects/Footage/fps/TestSeqFPS.mp4',
                      fourcc, 24, (1920, 1080))

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # check if the writer is None
    if i % 15 == 0:
        outputFrame = frame

    # write to output file
    out.write(outputFrame)

    i += 1


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()