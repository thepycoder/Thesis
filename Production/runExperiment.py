from Production import MobileNetDetector
from Production import HaarCascadeDetector
from Production import HogDetector
from Production import IouTracker
from Production import CountPeople
import os
import csv


def absoluteFilePaths(directory):
   for dirpath,_,filenames in os.walk(directory):
       for f in filenames:
           yield os.path.abspath(os.path.join(dirpath, f))

clipfolder = "/media/victor/57a90e07-058d-429d-a357-e755d0820324/Footage/Clips1/"
clipfolder = "/home/victor/Projects/Footage/Clips1"
f = open("../Results/test.csv", "w+")
reader = csv.writer(f, delimiter=',')

hog = HogDetector.HogDetector()
net = MobileNetDetector.MobileNetDetector(prototxt="../CNNs/MobileNetSSD_deploy.prototxt",
                                              caffemodel="../CNNs/MobileNetSSD_deploy.caffemodel",
                                              conf=0.4)
haar = HaarCascadeDetector.HaarCascadeDetector()
iou = IouTracker.IouTracker(treshold=0.3)

files = sorted(absoluteFilePaths(clipfolder), key=os.path.getsize)

for vid in files:
    print(vid.split('/')[-1])
    det = CountPeople.CountPeople(net, iou, 500)
    result = det.countInVideo(vid, showVideo=False)
    print(type(result[2]))
    reader.writerow([vid.split('/')[-1], result[0], result[1], result[2]])
    print("[RESULT] ", vid.split('/')[-1], result)
