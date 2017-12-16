import json
import csv
from pprint import pprint


f = open("Groundtruth/TestSeq1.json").read()
data = json.loads(f)

f = open("Groundtruth/gt.csv", "w")
reader = csv.writer(f)

pprint(data)

for key, value in data.items():
    frameNumber = key.split(".")[0]
    for k, v in data[key]['regions'].items():
        shape = data[key]['regions'][k]['shape_attributes']

        startX = int(shape['x'])
        startY = int(shape['y'])
        endX = startX + int(shape['width'])
        endY = startY + int(shape['height'])

        reader.writerow([frameNumber, startX, startY, endX, endY])