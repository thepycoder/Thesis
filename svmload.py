from imutils.object_detection import non_max_suppression
from imutils import resize
import numpy as np
import cv2

# svm = cv2.ml.SVM_load("SVMs/person.svm")
# sv = svm.getSupportVectors()
# retval, alpha, svidx = svm.getDecisionFunction(0)
#
# detector = np.vstack(sv[0])
# for i in sv[0]:
#     detector = np.append(detector, np.array([i]))

# detector = np.append(detector, np.array([0]))
# print(detector, detector[0], type(detector[0]))

# detector = sv

detector = np.loadtxt("SVMs/50-50.dat")
# print(detector, detector.shape)

image = "/home/victor/Projects/INRIAPerson/Train/pos/person_and_bike_125.png"
hog = cv2.HOGDescriptor()
# detector = hog.getDefaultPeopleDetector()
print(detector, detector[0], type(detector[0]))
# detector = detector[:-1]
hog.setSVMDetector(detector)

image = cv2.imread(image)
image = resize(image, width=min(400, image.shape[1]))
(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)

rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

# draw the final bounding boxes
for (xA, yA, xB, yB) in pick:
    cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

cv2.imshow("After NMS", image)
cv2.waitKey(0)