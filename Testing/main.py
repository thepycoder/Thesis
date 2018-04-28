import csv
import itertools
import random

from Testing import hogDetector

winstriderange = range(3, 8)
winstride = []

for i in range(len(winstriderange)):
    winstride += itertools.combinations(winstriderange, 2)

# print(winstride)

paddingrange = range(8, 33, 8)
padding = []

for i in range(len(paddingrange)):
    padding += itertools.combinations(paddingrange, 2)

# print(padding)

scalerange = range(80, 200, 1)
scale = [s / 100 for s in scalerange]

scale = [0.9, 1, 1.05, 1.4]

# print(scale)

combinations = [winstride, padding, scale]

params = list(itertools.product(*combinations))

f = open("results.csv", "w")
w = csv.writer(f)

random.shuffle(params)

index = 0
for i in params:
    index += 1
    hog = hogDetector.HogDetector(winstride=i[0], padding=i[1], scale=i[2])
    # hog.evaluateHog("/media/victor/57a90e07-058d-429d-a357-e755d0820324/Footage/TestSeq1.mp4", "Results/gt.csv")
    print("[INFO] Testing for winstride of %s, padding of %s and scale of %s" % (i[0], i[1], i[2]))
    fps, acc = hog.evaluateHog("../Footage/TestSeq1.mp4", "Results/gt.csv")
    w.writerow([fps, acc, i[0], i[1], i[2]])
    print("[INFO] %s rows" % index)
    print("-------------------------")
