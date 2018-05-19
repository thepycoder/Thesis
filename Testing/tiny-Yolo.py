import numpy as np
import time
import csv
import cv2


## Define the model parameters

cfg = "../Models/yolov3-tiny.cfg"
weights = "../Models/yolov3-tiny.weights"


## Set source file parameters and prepare the VideoCapture

# filename = "/home/victor/Projects/Footage/Clips1/00:00:30.442.mp4"
filename = "/media/victor/57a90e07-058d-429d-a357-e755d0820324/Footage/Clips1/00:00:30.442.mp4"

cap = cv2.VideoCapture()
cap.open(filename)
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


## Load the serialized model from disk

print("[INFO] loading model...")
net = cv2.dnn.readNetFromDarknet(cfg, weights)


## Start the timekeeping to calculate model speed

start = time.time()


## Main loop over the video file

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Prevent crash at the end of the video file
    if not ret:
        break

    # Get the frame dimensions for later on
    (h, w) = frame.shape[:2]

    # Create a blob from the source frame by resizing to the required 300x300 size
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (416, 416)), 1.0/255.0)

    # Feed blob to the net and perform a forward pass
    net.setInput(blob)
    detections = net.forward()

    print(detections)

    for i in range(detections.shape[0]):
        probability_index = 5
        probability_size = detections.shape[1] - probability_index

        objectClass = np.argmax(detections[i][probability_index:])
        confidence = detections[i][probability_index + objectClass]

        if confidence > 0.2:
            x_center = detections[i][0] * w
            y_center = detections[i][1] * h
            width = detections[i][2] * w
            height = detections[i][3] * h

            p1 = (int(x_center - width / 2), int(y_center - height / 2))
            p2 = (int(x_center + width / 2), int(y_center + height / 2))

            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)

    # loop over the detections
    # for i in np.arange(0, detections.shape[2]):
    #     # extract the confidence (i.e., probability) associated with the prediction
    #     confidence = detections[0, 0, i, 2]
    #
    #     # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
    #     if confidence > conf:
    #
    #         # Compute the (x, y)-coordinates of the bounding box for the object
    #         box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    #         (startX, startY, endX, endY) = box.astype("int")
    #
    #         # display the prediction
    #         label = "{:.2f}%".format(confidence * 100)
    #         cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
    #         y = startY - 15 if startY - 15 > 15 else startY + 15
    #         cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #
    # # Display the resulting frame
    # cv2.line(frame, (0, 360), (1280, 360), (0, 0, 0), 2)
    cv2.imshow('frame', frame)

    # Wait for q key to be pressed to stop the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calculate end time and print the speed benchmark results
end = time.time()
print("[INFO] it took %s seconds." % (end - start))
print("[INFO] clip has %s frames" % frameCount)
print("[INFO] that makes %s fps" % (frameCount / (end - start)))

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()