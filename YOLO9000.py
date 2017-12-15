import numpy as np
import time
import cv2

prototxt = "/media/victor/57a90e07-058d-429d-a357-e755d0820324/Projects/Tiny-yolo/yolo_tiny_deploy.prototxt"
model = "/media/victor/57a90e07-058d-429d-a357-e755d0820324/Projects/Tiny-yolo/yolo_tiny.caffemodel"
darknet = "/media/victor/57a90e07-058d-429d-a357-e755d0820324/Projects/darknet"
image = "/home/victor/Projects/Thesis/Testimages/test1.jpg"
conf = 0.1

from darknetpy.detector import Detector

detector = Detector(darknet,
                    '%s/cfg/coco.data' % darknet,
                    '%s/cfg/tiny-yolo.cfg' % darknet,
                    '%s/tiny-yolo.weights' % darknet)

start = time.time()
results = detector.detect(image)
end = time.time()
print("[INFO] it took %s seconds." % (end - start))


print(results)


# image = cv2.imread(image)
# (h, w) = image.shape[:2]
# # https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/
# blob = cv2.dnn.blobFromImage(cv2.resize(image, (448, 448)), 1, (448, 448), (104, 117, 123))
#
# # cv2.imshow("Output", image)
# # cv2.waitKey(0)
#
#
# # load our serialized model from disk
# print("[INFO] loading model...")
# net = cv2.dnn.readNetFromCaffe(prototxt, model)
#
#
# # pass the blob through the network and obtain the detections and
# # predictions
# print("[INFO] computing object detections...")
# start = time.time()
# net.setInput(blob)
# detections = net.forward()
# end = time.time()
# print("It took %s seconds." % (end - start))
#
#
# print(detections)

# loop over the detections
# for i in np.arange(0, detections.shape[2]):
#     # extract the confidence (i.e., probability) associated with the
#     # prediction
#     confidence = detections[0, 0, i, 2]
#
#     # filter out weak detections by ensuring the `confidence` is
#     # greater than the minimum confidence
#     if confidence > conf:
#         # extract the index of the class label from the `detections`,
#         # then compute the (x, y)-coordinates of the bounding box for
#         # the object
#         idx = int(detections[0, 0, i, 1])
#         box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#         (startX, startY, endX, endY) = box.astype("int")
#
#         # display the prediction
#         label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
#         print("[INFO] {}".format(label))
#         cv2.rectangle(image, (startX, startY), (endX, endY),
#                       COLORS[idx], 2)
#         y = startY - 15 if startY - 15 > 15 else startY + 15
#         cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
#
#
# cv2.imshow("Output", image)
# cv2.waitKey(0)