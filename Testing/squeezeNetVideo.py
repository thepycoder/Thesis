import numpy as np
import time
import cv2

prototxt = "../Models/SqueezeNetSSD.prototxt"
model = "../Models/SqueezeNetSSD.caffemodel"
image = "../Testimages/test15.png"
conf = 0.1


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

cap = cv2.VideoCapture()
vid = cap.open("../../Footage/TestSeq1.mp4")
# vid = cap.open("/media/victor/57a90e07-058d-429d-a357-e755d0820324/Footage/TestSeq1.mp4")
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

ret, frame = cap.read()
wFrame = frame.shape

print(wFrame)

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
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1, (300, 300), 127.5)
    # blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.003, (300, 300), 127.5)

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
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)


    # Display the resulting frame
    cv2.line(frame, (0, 360), (1280, 360), (255, 255, 255))
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end = time.time()
print("[INFO] it took %s seconds." % (end - start))
print("[INFO] clip has %s frames" % frameCount)
print("[INFO] that makes %s fps" % (frameCount / (end - start)))

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()