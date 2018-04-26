import random
import cv2
import os

negative_path = "/media/victor/57a90e07-058d-429d-a357-e755d0820324/INRIA/INRIAPerson/train_64x128_H96/neg/"
output_path = "/media/victor/57a90e07-058d-429d-a357-e755d0820324/INRIA/Testing/neg/"


def divide_equally(n):
    if n < 3:
        return [n, 1]
    result = list()
    for i in range(1, int(n ** 0.5) + 1):
        div, mod = divmod(n, i)
        # ignore 1 and n itself as factors
        if mod == 0 and i != 1 and div != n:
            result.append(div)
            result.append(i)
    if len(result) == 0:  # if no factors then add 1
        return divide_equally(n + 1)
    return result[len(result) - 2:]


for i in os.listdir(negative_path)[1:10]:
    image = cv2.imread(negative_path + i)
    print(image.shape)  # width, height, color

    W = int(image.shape[0] / 64)
    H = int(image.shape[1] / 128)

    print(divide_equally(W*H))
