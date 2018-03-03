import numpy as np
import cv2
import os

positive_path = "/media/victor/57a90e07-058d-429d-a357-e755d0820324/INRIA/INRIAPerson/train_64x128_H96/pos/"
output_path = "/media/victor/57a90e07-058d-429d-a357-e755d0820324/INRIA/Testing/pos/"

for i in os.listdir(positive_path)[1:301]:
    image = cv2.imread(positive_path + i)
    w = image.shape[1]
    h = image.shape[0]

    crop = image[16:h-16, 16:w-16]  # INRIA website states 16px margin on each side
    cv2.imwrite(output_path + i, crop)

