from imutils.object_detection import non_max_suppression
from imutils import resize
import numpy as np
import time
import cv2


image = "Testimages/test1.jpg"

hog = cv2.HOGDescriptor()
# hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# svm = cv2.ml.SVM_load("/home/victor/Projects/trainHOG/genfiles/descriptorvector.dat")
# sv = svm.getSupportVectors()
# rho, alpha, svidx = svm.getDecisionFunction(0)
# svm = cv2.ml.SVM_create()
hog.load("/home/victor/Projects/trainHOG/genfiles/cvHOGClassifier.yaml")

# print(svm)
#
# hog.setSVMDetector(svm)

# load the image and resize it to (1) reduce detection time
# and (2) improve detection accuracy
image = cv2.imread(image)
image = resize(image, width=min(400, image.shape[1]))
orig = image.copy()

# detect people in the image
start = time.time()
(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.5)
end = time.time()
print("It took %s seconds." % (end - start))

# apply non-maxima suppression to the bounding boxes using a
# fairly large overlap threshold to try to maintain overlapping
# boxes that are still people
rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

# draw the final bounding boxes
for (xA, yA, xB, yB) in pick:
    cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

cv2.imshow("After NMS", image)
cv2.waitKey(0)