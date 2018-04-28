import csv

import cv2

from Testing import iou

groundtruth = open("Results/gt.csv")
detections = open("Results/results.csv")

gt = csv.reader(groundtruth)
det = csv.reader(detections)

gt_list = []
det_list = []

threshHold = 0.5

goodDetections = 0
badDetections = 0

for row in gt:
    gt_list.append(row)

for row in det:
    det_list.append(row)

for bbox1 in det_list:
    highestIOU = 0
    for bbox2 in gt_list:
        if bbox1[0] == bbox2[0]:
            v = iou.iou(bbox1, bbox2)
            bbox1 = [int(x) for x in bbox1]
            bbox2 = [int(x) for x in bbox2]
            image = cv2.imread("/media/victor/57a90e07-058d-429d-a357-e755d0820324/Footage/TestSeq1/%s.jpg" % bbox1[0])
            cv2.rectangle(image, (bbox1[1], bbox1[2]), (bbox1[3], bbox1[4]), (102, 255, 0), 2)
            cv2.rectangle(image, (bbox2[1], bbox2[2]), (bbox2[3], bbox2[4]), (0, 0, 0), 2)
            cv2.putText(image, "IOU: %s" % v, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.imshow('frame', image)
            if cv2.waitKey(150) & 0xFF == ord('q'):
                break
            if v > highestIOU:
                highestIOU = v
    if highestIOU > threshHold:
        goodDetections += 1
    else:
        badDetections += 1

print("The amount of good detections was: %s" % goodDetections)
print("The amount of missed detections was %s" % badDetections)
print("The total amount of detections was: %s while the groundtruth had %s detections" % (goodDetections+badDetections, gt_list.__len__()))
print("The total detection accuracy was: %s" % ((goodDetections / (gt_list.__len__())) * 100))
