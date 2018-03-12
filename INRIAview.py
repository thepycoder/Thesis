import codecs
import re
import cv2

filename = "crop001038.png"
image_path = '/home/victor/Projects/INRIAPerson/Train/pos/'
anno_path = '/home/victor/Projects/INRIAPerson/Train/annotations/'
image = cv2.imread(image_path + filename)

f = codecs.open(anno_path + filename.split('.')[0] + '.txt', encoding='ISO-8859-14')
for line in f.readlines():
    if not line.startswith('#'):  # are comments
        if line.startswith('Bounding box'):
            bbox = re.findall(r'\((.*?)\)', line)[2:]
            Xmin = bbox[0].split(', ')[0]
            Ymin = bbox[0].split(', ')[1]
            Xmax = bbox[1].split(', ')[0]
            Ymax = bbox[1].split(', ')[1]

            width = int(Xmax) - int(Xmin)
            height = int(Ymax) - int(Ymin)

            cv2.rectangle(image, (int(Xmin), int(Ymin)), (int(Xmax), int(Ymax)), (255, 0, 0))


cv2.imshow('with', image)
cv2.waitKey(0)