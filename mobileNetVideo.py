import numpy as np
import time
import csv
import cv2


## Define the model parameters

prototxt = "CNNs/MobileNetSSD_deploy.prototxt"
model = "CNNs/MobileNetSSD_deploy.caffemodel"
conf = 0.4


## Set source file parameters and prepare the VideoCapture

filename = "/media/victor/57a90e07-058d-429d-a357-e755d0820324/Footage/Clips1/00:13:57.166.mp4"

cap = cv2.VideoCapture()
vid = cap.open(filename)
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


## Load the serialized model from disk

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)


## Start the timekeeping to calculate model speed

start = time.time()


## Create empty dict to dump detections into so we can evaluate the performance

f = open("Results/results.csv", "w")
reader = csv.writer(f)
frameNumber = 0


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
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    # Feed blob to the net and perform a forward pass
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
        if confidence > conf:

            # Compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # display the prediction
            label = "{:.2f}%".format(confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # add the detection to the csv file
            reader.writerow([frameNumber, startX, startY, endX, endY])

    # Display the resulting frame
    cv2.line(frame, (0, 360), (1280, 360), (0, 0, 0), 2)
    cv2.imshow('frame', frame)
    cv2.imshow('blob', cv2.dnn.imagesFromblob(blob))

    # Wait for q key to be pressed to stop the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Finally update the framenumber, so the csv writer knows at which frame we are
    frameNumber += 1

# Calculate end time and print the speed benchmark results
end = time.time()
print("[INFO] it took %s seconds." % (end - start))
print("[INFO] clip has %s frames" % frameCount)
print("[INFO] that makes %s fps" % (frameCount / (end - start)))

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()