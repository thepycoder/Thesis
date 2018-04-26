import MobileNetDetector
import HaarCascadeDetector
import YoloDetector
import HogDetector
import IouTracker
import CountPeople
import os
import csv


def absoluteFilePaths(directory):
   for dirpath,_,filenames in os.walk(directory):
       for f in filenames:
           yield os.path.abspath(os.path.join(dirpath, f))


# clipfolder = "/media/victor/57a90e07-058d-429d-a357-e755d0820324/Footage/Clips1/"
clipfolder = "../../Footage/Clips1"

hog_slow = HogDetector.HogDetector()
hog_fast = HogDetector.HogDetector(winstride=(16, 16), padding=(4, 4))
net = MobileNetDetector.MobileNetDetector(prototxt="../Models/MobileNetSSD_deploy.prototxt",
                                              caffemodel="../Models/MobileNetSSD_deploy.caffemodel",
                                              conf=0.4)
haar_upper = HaarCascadeDetector.HaarCascadeDetector(classifierfile="../Models/haarcascade_upperbody.xml")
haar_full = HaarCascadeDetector.HaarCascadeDetector(classifierfile="../Models/haarcascade_fullbody.xml")
yolo = YoloDetector.YoloDetector(cfg="../Models/yolov2-tiny.cfg",
                                     weights="../Models/yolov2-tiny.weights",
                                     conf=0.3)
iou = IouTracker.IouTracker(treshold=0.3)

detectors = [haar_upper, haar_full, hog_slow, hog_fast, yolo, net]

files = sorted(absoluteFilePaths(clipfolder), key=os.path.getsize)

for detector in detectors:

    f = open("../Results/%s.csv" % detector.getName(), "w+")
    reader = csv.writer(f, delimiter=',')
    reader.writerow(['File', 'UP', 'DOWN', 'FPS'])

    index = 1
    for vid in files:
        # print(vid.split('/')[-1])
        det = CountPeople.CountPeople(detector, iou, 440)
        result = det.countInVideo(vid, showVideo=False)
        # print(type(result[2]))
        reader.writerow([vid.split('/')[-1], result[0], result[1], result[2]])
        # print("[RESULT] ", vid.split('/')[-1], result)
        print("[PROGRESS] ", index)
        index += 1
