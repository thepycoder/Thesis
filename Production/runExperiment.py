from multiprocessing.dummy import Pool as ThreadPool
import MobileNetDetector
import HaarCascadeDetector
import SqueezeNetDetector
import YoloDetector
import HogDetector
import numpy as np
import IouTracker
import CountPeople
import os
import csv


def absoluteFilePaths(directory):
   for dirpath,_,filenames in os.walk(directory):
       for f in filenames:
           yield os.path.abspath(os.path.join(dirpath, f))


# make the Pool of workers
pool = ThreadPool(8)

# clipfolder = "/media/victor/57a90e07-058d-429d-a357-e755d0820324/Footage/Clips1/"
clipfolder = "../../Footage/Clips1"

hog = HogDetector.HogDetector(name='hog')
hog_50 = HogDetector.HogDetector(name='hog_50', svmdetector=np.loadtxt("../SVMs/50-50.dat"))
# hog_ALL = HogDetector.HogDetector(name='hog_ALL', svmdetector=np.loadtxt("../SVMs/ALL.dat"))
hog_scaled = HogDetector.HogDetector(name='hog_scaled')
net = MobileNetDetector.MobileNetDetector(prototxt="../Models/MobileNetSSD_deploy.prototxt",
                                              caffemodel="../Models/MobileNetSSD_deploy.caffemodel",
                                              conf=0.2)
haar_upper = HaarCascadeDetector.HaarCascadeDetector(classifierfile="../Models/haarcascade_upperbody.xml", name='haar_upper')
haar_full = HaarCascadeDetector.HaarCascadeDetector(classifierfile="../Models/haarcascade_fullbody.xml", name='haar_full')
yolo = YoloDetector.YoloDetector(cfg="../Models/yolov2-tiny.cfg",
                                     weights="../Models/yolov2-tiny.weights",
                                     conf=0.2)
squeeze = SqueezeNetDetector.SqueezeNetDetector(prototxt="../Models/SqueezeNetSSD.prototxt",
                                              caffemodel="../Models/SqueezeNetSSD.caffemodel",
                                              conf=0.2)
iou = IouTracker.IouTracker(treshold=0.3)

detectors = [net, yolo, squeeze]

files = sorted(absoluteFilePaths(clipfolder), key=os.path.getsize)

for detector in detectors:
# def processDetector(detector):
    f = open("../Results/%s.csv" % detector.getName(), "w+")
    reader = csv.writer(f, delimiter=',')
    reader.writerow(['File', 'UP', 'DOWN', 'FPS'])

    index = 1
    for vid in files:
        # print(vid.split('/')[-1])
        det = CountPeople.CountPeople(detector, iou, 440, 0)
        result = det.countInVideo(vid, showVideo=False, showSpeed=False)
        # print(type(result[2]))
        reader.writerow([vid.split('/')[-1], result[0], result[1], result[2]])
        # print("[RESULT] ", vid.split('/')[-1], result)
        print("[PROGRESS] ", detector.getName(), index)
        index += 1
