import numpy as np
import cv2
import os

negative_path = "/media/victor/57a90e07-058d-429d-a357-e755d0820324/INRIA/INRIAPerson/train_64x128_H96/neg/"
output_path = "/media/victor/57a90e07-058d-429d-a357-e755d0820324/INRIA/Testing/neg/"

for i in os.listdir(negative_path)[1:1001]:
    image = cv2.imread(negative_path + i)
    w = image.shape[1]
    h = image.shape[0]

    x = np.random.randint(w-64, size=10)
    y = np.random.randint(h-128, size=10)

    for j in range(10):
        crop = image[y[j]:y[j] + 128, x[j]:x[j] + 64]

        cv2.imwrite(output_path + str(j) + i, crop)

        cv2.rectangle(image, (x[j], y[j]), (x[j]+64, y[j]+128), (255, 0, 0))

    #cv2.rectangle(image, (1,1), (w, h), (0, 255, 0))