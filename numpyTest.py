import numpy as np
import time


s1 = time.time()

test = np.array([[0, 0, 0, 0]])

el1 = np.array([[1, 2, 3, 4]])
el2 = np.array([[5, 6, 7, 8]])
test = np.concatenate((test, el1))
test = np.concatenate((test, el2))

e1 = time.time()

print(test, e1-s1)


s2 = time.time()

test2 = np.array([[0, 0, 0, 0]])

el1 = np.array([[1, 2, 3, 4]])
el2 = np.array([[5, 6, 7, 8]])
test2 = np.append(test2, el1, axis=0)
test2 = np.append(test2, el2, axis=0)

e2 = time.time()

print(test, e2-s2)

print(test)
