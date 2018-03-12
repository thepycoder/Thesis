from pycocotools.coco import COCO
import cv2
import sys
import pylab

pylab.rcParams['figure.figsize'] = (8.0, 10.0)

dataDir = "/home/victor/Projects/COCO/"
dataType = "train2014"
annFile = "{}/annotations/instances_{}.json".format(dataDir, dataType)

coco = COCO(annFile)

# display COCO categories and supercategories
# cats = coco.loadCats(coco.getCatIds())
# nms = [cat['name'] for cat in cats]
# print('COCO categories: \n{}\n'.format(' '.join(nms)))
#
# nms = set([cat['supercategory'] for cat in cats])
# print('COCO supercategories: \n{}'.format(' '.join(nms)))

catIds = coco.getCatIds(catNms=['person'])
imgIds = coco.getImgIds(catIds=catIds)
l = len(imgIds)

for i in range(1, l+1):

    sys.stdout.write("Progress: %d / %d   \r" % (i, l))
    sys.stdout.flush()

    img = coco.loadImgs(imgIds[i])[0]

    image = cv2.imread(dataDir + dataType + '/' + img['file_name'])

    annIds = coco.getAnnIds(imgIds=img['id'])
    anns = coco.loadAnns(annIds)

    # cv2.imshow("Image", image)
    # cv2.waitKey(0)

    crop_id = 0

    for ann in anns:
        if ann['category_id'] == 1:
            bbox = ann['bbox']
            x = int(bbox[0])
            y = int(bbox[1])
            w = int(bbox[2])
            h = int(bbox[3])
            # cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

            crop = image[y:y + h, x:x + w]

            if h == 0 or w == 0:
                print("Anomaly: " + img['file_name'])
                break

            cropSized = cv2.resize(crop, (64, 128))
            (hc, wc) = cropSized.shape[:2]

            # if hc < 128:
            #     cropSized = resize(crop, width=128, height=64)
            #
            # if hc >= 128:
            #     cropSized = cropSized[0:64, 0:128]

            cv2.imwrite(dataDir + "pos/" + str(crop_id) + "_" + img['file_name'], cropSized)

            # print(ann['area'])

            # cv2.imshow("Cropped Image", crop)
            # cv2.waitKey(0)

            crop_id += 1


