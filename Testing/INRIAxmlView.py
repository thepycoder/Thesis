import xml.etree.ElementTree as ET
import codecs
import re
import cv2

  # /home/victor/Projects/INRIAPerson/Train/pos/crop001160.png
  # /home/victor/Projects/INRIAPerson/Train/pos/crop001038.png
  # /home/victor/Projects/INRIAPerson/Train/pos/crop001612.png
  # /home/victor/Projects/INRIAPerson/Train/pos/crop001709.png
  # /home/victor/Projects/INRIAPerson/Train/pos/crop001002.png
  # /home/victor/Projects/INRIAPerson/Train/pos/crop001027.png

filename = "crop001027.png"
image_path = '/home/victor/Projects/INRIAPerson/Train/pos/'
tree = ET.parse('Dataprep/annotations.xml')
frame = cv2.imread(image_path + filename)

for image in tree.find('images'):
    if image.attrib['file'] == image_path + filename:
        for box in image.findall('box'):
            print(box.attrib)
            cv2.rectangle(frame,
                          (int(box.attrib['left']), int(box.attrib['top'])),
                          (int(box.attrib['left']) + int(box.attrib['width']), int(box.attrib['top']) + int(box.attrib['height'])),
                          (0, 255, 0))
            cv2.circle(frame, (int(box.attrib['left']), int(box.attrib['top'])), 5, (0, 0, 255))


cv2.imshow('test', frame)
cv2.waitKey()
