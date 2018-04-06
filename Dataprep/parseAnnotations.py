import json
import csv
from pprint import pprint


f = open("Results/TestSeq1.json").read()
data = json.loads(f)

f = open("Results/gt.csv", "w")
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