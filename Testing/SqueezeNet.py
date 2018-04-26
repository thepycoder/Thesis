import numpy as np
import time
import csv
import cv2


## Define the model parameters

prototxt = "/home/victor/Projects/Thesis/Models/SqueezeNet.prototxt"
model = "/home/victor/Projects/Thesis/Models/squeezenet_v1.1.caffemodel"
conf = 0.4


## Set source file parameters and prepare the VideoCapture
filename = "/home/victor/Projects/Footage/Clips1/00:00:30.442.mp4"

cap = cv2.VideoCapture()
cap.open(filename)
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


## Load the serialized model from disk

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)


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
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (227, 227)), 1.0 / 255.0)

    # Feed blob to the net and perform a forward pass
    net.setInput(blob)
    detections = net.forward()

    print(detections)

    # loop over the detections

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