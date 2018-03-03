import cv2
import glob
import os
import numpy as np

hog = cv2.HOGDescriptor()

samples = []
labels = []

# positive_path = "/home/victor/Projects/COCO/trainPOS/"
# negative_path = "/home/victor/Projects/COCO/trainNEG/"

positive_path = "/media/victor/57a90e07-058d-429d-a357-e755d0820324/INRIA/Testing/pos/"
negative_path = "/media/victor/57a90e07-058d-429d-a357-e755d0820324/INRIA/Testing/neg/"

print("Processing positive samples")
# Get positive samples
for filename in os.listdir(positive_path):
    img = cv2.imread(positive_path + filename, 1)
    hist = hog.compute(img)
    samples.append(hist)
    labels.append(1)

print("Processing negative samples")
# Get negative samples
for filename in os.listdir(negative_path):
    img = cv2.imread(negative_path + filename, 1)
    hist = hog.compute(img)
    samples.append(hist)
    labels.append(0)

# Convert objects to Numpy Objects
samples = np.float32(samples)
labels = np.array(labels)


# Shuffle Samples
rand = np.random.RandomState(321)
shuffle = rand.permutation(len(samples))
samples = samples[shuffle]
labels = labels[shuffle]

print("Training the classifier")
# Create SVM classifier
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR) # cv2.ml.SVM_LINEAR
# svm.setDegree(0.0)
svm.setGamma(5.383)
# svm.setCoef0(0.0)
svm.setC(2.67)
# svm.setNu(0.0)
# svm.setP(0.0)
# svm.setClassWeights(None)

# Train
svm.trainAuto(samples, cv2.ml.ROW_SAMPLE, labels)
svm.save('svm_data.dat')
print("Done!")
