import itertools
import hogDetector


winstriderange = range(3, 8)
winstride = []

for i in range(len(winstriderange)):
    winstride += itertools.combinations(winstriderange, 2)

print(winstride)

paddingrange = range(8, 33, 8)
padding = []

for i in range(len(paddingrange)):
    padding += itertools.combinations(paddingrange, 2)

print(padding)

scalerange = range(80, 200, 1)
scale = [s / 100 for s in scalerange]

scale = [0.9, 1, 1.05, 1.4]

print(scale)

combinations = [winstride, padding, scale]

params = list(itertools.product(*combinations))

print(params)

for i in params:
    hog = hogDetector.HogDetector(winstride=params[0][0], padding=params[0][1], scale=params[0][2])
    hog.evaluateHog("/media/victor/57a90e07-058d-429d-a357-e755d0820324/Footage/TestSeq1.mp4", "Groundtruth/gt.csv")