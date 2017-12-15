from scipy import spatial
import numpy as np
import time
import cv2

prototxt = "MobileNetSSD_deploy.prototxt"
model = "MobileNetSSD_deploy.caffemodel"
conf = 0.5
lineHeight = 360

# initialise the counting line array
C = np.array([[0, 0, 0, 0]])
M = np.array([[0, 0]])
Cold = C
Mold = M

UP = 0
DOWN = 0


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

cap = cv2.VideoCapture()
# vid = cap.open("../Footage/TestSeq7.mp4")
vid = cap.open("/media/victor/57a90e07-058d-429d-a357-e755d0820324/Footage/TestSeq1.mp4")
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# ret, frame = cap.read()

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)

start = time.time()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    # Our operations on the frame come here
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > conf:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # display the prediction
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            # print("[INFO] {}".format(label))
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)

            # pass the detection through to the counting algo if the rectangle crosses the counting line
            if startY < lineHeight < endY:
                midX = (startX + endX) / 2
                midY = (startY + endY) / 2
                M = np.append(M, np.array([[midX, midY]]), axis=0)
                C = np.append(C, np.array([[startX, startY, endX, endY]]), axis=0)

            # put text with class name and confidence value
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    # here we count the people using our algorithm
    if C.shape[0] > Cold.shape[0]:
        # new detection in counting line
        # first find out which one they are
        nrOfNewDets = C.shape[0] - Cold.shape[0]
        # calculate pairwise distances
        distances = spatial.distance.cdist(Mold, M)
        distances = np.amin(distances, axis=0)
        # determine the new detections by them having no previous point -> far from the old values
        # furthest holds the indexes of the new detections
        furthest = distances.argsort()[-nrOfNewDets:][::-1]

        # now to determine the direction of the new detections
        # we look at where the bbox crosses the counting line
        # if it is on the lower half -> going down
        # if it is on the upper half -> going up
        for i in furthest:
            # if LOI is between top and middle of bbox
            if C[i, 1] < lineHeight < M[i, 1]:
                UP += 1
            else:
                DOWN += 1

    Cold = C
    Mold = M
    C = np.array([[0, 0, 0, 0]])
    M = np.array([[0, 0]])

    # Display the resulting frame
    cv2.line(frame, (0, lineHeight), (1280, lineHeight), (255, 255, 255))
    cv2.putText(frame, "UP: %s" % UP, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[15], 2)
    cv2.putText(frame, "DOWN: %s" % DOWN, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[15], 2)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


end = time.time()
print("[INFO] it took %s seconds." % (end - start))
print("[INFO] clip has %s frames" % frameCount)
print("[INFO] that makes %s fps" % (frameCount / (end - start)))

print("[RESULT] the amount of people going up was: %s" % UP)
print("[RESULT] the amount of people going down was: %s" % DOWN)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()