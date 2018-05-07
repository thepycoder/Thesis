import cv2
import random
import os
import pickle
import itertools
import imutils
import numpy as np
from Production import HogDetector
from sklearn.svm import LinearSVC
import skimage.feature
from skimage.transform import pyramid_gaussian
from sklearn.externals import joblib

# positive_path = "/home/victor/Projects/COCO/trainPOS/"
# negative_path = "/home/victor/Projects/COCO/trainNEG/"

# positive_path = "/media/victor/57a90e07-058d-429d-a357-e755d0820324/INRIA/Testing/pos/"
# negative_path = "/media/victor/57a90e07-058d-429d-a357-e755d0820324/INRIA/Testing/neg/"
# hard_negative_path = "/media/victor/57a90e07-058d-429d-a357-e755d0820324/INRIA/Testing/hardnegatives/"
positive_path = "/home/victor/Projects/INRIAPerson/NEWTRAINING/pos/"
negative_path = "/home/victor/Projects/INRIAPerson/NEWTRAINING/allneg/"
hard_negative_path = "/home/victor/Projects/INRIAPerson/NEWTRAINING/hardnegatives/"

random.seed(42)


def prepare_samples(positive_path, negative_path):
    hog = cv2.HOGDescriptor()

    samples = []
    labels = []

    print("Processing positive samples")
    # Get positive samples
    for filename in os.listdir(positive_path):
        img = cv2.imread(positive_path + filename, 1)
        hist = list(itertools.chain.from_iterable(hog.compute(img)))
        # print(hist)
        samples.append(hist)
        labels.append(1)

    print("Processing negative samples")
    # Get negative samples (as many as the positive samples)
    for filename in os.listdir(negative_path):
        img = cv2.imread(negative_path + filename, 1)
        hist = list(itertools.chain.from_iterable(hog.compute(img)))
        samples.append(hist)
        labels.append(0)

    return samples, labels


def train_hog(samples, labels):
    # Convert objects to Numpy Objects
    samples = np.array(samples, 'float64')
    labels = np.array(labels)

    # Shuffle Samples
    rand = np.random.RandomState(321)
    shuffle = rand.permutation(len(samples))
    samples = samples[shuffle]
    labels = labels[shuffle]

    print("Training the classifier")
    clf = LinearSVC(class_weight='balanced')
    clf.fit(samples, labels)

    f = open('../SVMs/svmlight_no_hardneg.dat', 'w')
    for value in clf.coef_[0]:
        f.write('%s ' % value)
    f.close()

    return clf


def pyramid(image, scale=1.5, minSize=(30, 30)):
    # yield the original image
    yield image

    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        # yield the next image in the pyramid
        yield image


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def hardnegmining(hard_negative_path, samples, labels, clf):
    winW = 64
    winH = 128
    hog = cv2.HOGDescriptor()
    hardneg_samples = []
    hardneg_labels = []
    # Hard negative mining
    print("Starting hard negative mining")

    detector = HogDetector.HogDetector(name='hog', svmdetector=np.loadtxt("../SVMs/svmlight_no_hardneg.dat"))

    for filename in os.listdir(hard_negative_path):
        print(filename)
        img = cv2.imread(hard_negative_path + filename, 1)
        # for (i, resized) in enumerate(pyramid_gaussian(img, downscale=2)):
        for resized in pyramid(img, scale=1.05):
            for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
                # if the window does not meet our desired window size, ignore it
                if window.shape[0] != winH or window.shape[1] != winW:
                    continue
                hist = [list(itertools.chain.from_iterable(hog.compute(window)))]
                classification = clf.predict(hist)
                if classification == 1:
                    # hardneg_samples.append(hist[0])
                    # hardneg_labels.append(0)
                    cv2.imwrite("/home/victor/Projects/INRIAPerson/NEWTRAINING/hardnegcrops/" + str(int(x)) + str(int(y)) + filename, window)

    return hardneg_samples, hardneg_labels


print("Started!")
samples, labels = prepare_samples(positive_path, negative_path)
clf = train_hog(samples, labels)
# hardneg_samples, hardneg_labels = hardnegmining(hard_negative_path, samples, labels, clf)
# final_samples = samples + hardneg_samples
# final_labels = labels + hardneg_labels
# final_clf = train_hog(final_samples, final_labels)
print("Done!")
