import cv2
import csv


def evalutate(groundtruth, detections, threshhold):
    f = open(groundtruth)
    gt = csv.reader(f)

    gt_list = []

    gooddetections = 0
    baddetections = 0

    for row in gt:
        gt_list.append([int(i) for i in row])

    for bbox1 in detections:
        highestIOU = 0
        for bbox2 in gt_list:
            if bbox1[0] == bbox2[0]:
                v = iou(bbox1, bbox2)
                if v > highestIOU:
                    highestIOU = v
        if highestIOU > threshhold:
            gooddetections += 1
        else:
            baddetections += 1

    print("The amount of good detections was: %s" % gooddetections)
    print("The amount of missed detections was %s" % baddetections)
    print("The total amount of detections was: %s while the groundtruth had %s detections"
          % (gooddetections + baddetections, gt_list.__len__()))
    print("The total detection accuracy was: %s" % ((gooddetections / (gt_list.__len__())) * 100))


def iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.

    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
        bbox2 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.

    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """

    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (frameNr, x0_1, y0_1, x1_1, y1_1) = bbox1
    (frameNr, x0_2, y0_2, x1_2, y1_2) = bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union