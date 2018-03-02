from pycocotools.coco import COCO
from random import randint
import cv2
import sys
import pylab

pylab.rcParams['figure.figsize'] = (8.0, 10.0)

dataDir = "/home/victor/Projects/COCO/"
dataType = "train2014"
annFile = "{}/annotations/instances_{}.json".format(dataDir, dataType)

coco = COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
# print('COCO categories: \n{}\n'.format(' '.join(nms)))
#
# nms = set([cat['supercategory'] for cat in cats])
# print('COCO supercategories: \n{}'.format(' '.join(nms)))

catIds = coco.getCatIds(catNms=['person'])
imgIdsPerson = coco.getImgIds(catIds=catIds)

imgIdsAll = coco.getImgIds()

imgIds = [x for x in imgIdsAll if x not in imgIdsPerson]

l = len(imgIds)

for i in range(1, l):

    sys.stdout.write("Progress: %d / %d   \r" % (i, l))
    sys.stdout.flush()

    img = coco.loadImgs(imgIds[i])[0]

    image = cv2.imread(dataDir + dataType + '/' + img['file_name'])
    (h, w) = image.shape[:2]

    if h > 128 and w > 64:

        for j in range(1, 11):
            x = randint(0, w-64)
            y = randint(0, h-128)

            crop = image[y:y+128, x:x+64]

            cv2.imwrite(dataDir + "neg/" + str(j) + "_" + img['file_name'], crop)

