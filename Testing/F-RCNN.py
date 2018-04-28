import numpy as np
import time
import cv2

# prototxt = "F-RCNN/bvlc_googlenet.prototxt"
model = "F-RCNN/frozen_inference_graph.pb"
image = "Testimages/test7.png"
# labels = "RCNN/synset_words.txt"
conf = 0.1


# load the class labels from disk
# rows = open(labels).read().strip().split("\n")
# classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]


image = cv2.imread(image)
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(image, 1, (1200, 1200), (104, 117, 123))

# cv2.imshow("Output", image)
# cv2.waitKey(0)


# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromTensorflow(model)


# pass the blob through the network and obtain the detections and
# predictions
print("[INFO] computing object detections...")
start = time.time()
net.setInput(blob)
detections = net.forward()
end = time.time()
print("It took %s seconds." % (end - start))


# sort the indexes of the probabilities in descending order (higher
# probabilitiy first) and grab the top-5 predictions
# idxs = np.argsort(detections[0])[::-1][:5]
#
# # loop over the top-5 predictions and display them
# for (i, idx) in enumerate(idxs):
#     # draw the top prediction on the input image
#     if i == 0:
#         text = "Label: {}, {:.2f}%".format(classes[idx],
#                                            detections[0][idx] * 100)
#         cv2.putText(image, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX,
#                     0.7, (0, 0, 255), 2)
#
#     # display the predicted label + associated probability to the
#     # console
#     print("[INFO] {}. label: {}, probability: {:.5}".format(i + 1, classes[idx], detections[0][idx]))
#
#
# cv2.imshow("Output", image)
# cv2.waitKey(0)