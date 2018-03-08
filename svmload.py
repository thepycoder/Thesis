from imutils.object_detection import non_max_suppression
from imutils import resize
import numpy as np
import cv2

svm = cv2.ml.SVM_load("300iter.dat")
sv = svm.getSupportVectors()
retval, alpha, svidx = svm.getDecisionFunction(0)

print("C:", svm.getC(),
      "Gamma: ", svm.getGamma(),
      "P: ", svm.getP(),
      "NU: ", svm.getNu(),
      "Coef0: ", svm.getCoef0(),
      "Degree: ", svm.getDegree())

detector = np.append(sv, retval)

image = "Testimages/test9.png"
hog = cv2.HOGDescriptor()
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