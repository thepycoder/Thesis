import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog
import cv2


plt.axis('off')

# Read image
img = cv2.imread('../Testimages/tine.png')
img = np.float32(img) / 255.0


# cv2.imshow('frame', img)
plt.imshow(img)
plt.show()
cv2.waitKey()

# Calculate gradient
gx = cv2.Sobel(img, -1, 1, 0, ksize=1)
gy = cv2.Sobel(img, -1, 0, 1, ksize=1)



# cv2.imshow('x', gx)
plt.axis('off')
plt.imshow(gx)
plt.show()
cv2.waitKey()
# cv2.imwrite('../Testimages/gradient_x.png', gx)
# cv2.imshow('y', gy)
# cv2.imwrite('../Testimages/gradient_y.png', gy)
plt.axis('off')
plt.imshow(gy)
plt.show()
cv2.waitKey()

# Python Calculate gradient magnitude and direction ( in degrees )
mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

# cv2.imshow('mag', mag)
# cv2.imwrite('../Testimages/gradient_magnitude.png', mag)
plt.axis('off')
plt.imshow(mag)
plt.show()
cv2.waitKey()

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

fd, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True)

plt.axis('off')
plt.imshow(hog_image, cmap=plt.cm.gray)
plt.show()