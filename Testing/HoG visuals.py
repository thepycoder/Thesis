import cv2
import numpy as np

# Python gradient calculation

# Read image
img = cv2.imread('../Testimages/tine.png')
img = np.float32(img) / 255.0

cv2.imshow('frame', img)
cv2.waitKey()

# Calculate gradient
gx = cv2.Sobel(img, -1, 1, 0, ksize=1)
gy = cv2.Sobel(img, -1, 0, 1, ksize=1)

cv2.imshow('x', gx)
cv2.imwrite('../Testimages/gradient_x.png', gx)
cv2.imshow('y', gy)
cv2.imwrite('../Testimages/gradient_y.png', gy)
cv2.waitKey()

# Python Calculate gradient magnitude and direction ( in degrees )
mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

cv2.imshow('mag', mag)
cv2.imwrite('../Testimages/gradient_magnitude.png', mag)
cv2.waitKey()

