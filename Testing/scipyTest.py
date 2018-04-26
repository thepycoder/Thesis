from scipy import spatial
import numpy as np

A = np.random.random((10, 2))*100
B = np.random.random((5, 2))*100

test = spatial.distance.cdist(A, B)
print(test)

test2 = np.amin(test, axis=0)
print(test2)

outlier = np.argmax(test2)
print(outlier)

print(test2.argsort()[-3:][::-1])
