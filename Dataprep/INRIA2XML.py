import xml.etree.cElementTree as ET
import codecs
import os
import re
import math


annotations = '/home/victor/Projects/INRIAPerson/Train/annotations/'
image_path = '/home/victor/Projects/INRIAPerson/Train/pos/'

ds = ET.Element('dataset')
images = ET.SubElement(ds, 'images')

aspect_ratios = []

for a in os.listdir(annotations):
    f = codecs.open(annotations + a, encoding='ISO-8859-14')
    image = ET.SubElement(images, 'image')
    for line in f.readlines():
        if not line.startswith('#'):  # are comments
            if line.startswith('Image filename'):
                name = re.findall(r'"(.*?)"', line)[0].split('/')[-1]
                image.attrib['file'] = image_path + name
            elif line.startswith('Bounding box'):
                bbox = re.findall(r'\((.*?)\)', line)[2:]
                Xmin = bbox[0].split(', ')[0]
                Ymin = bbox[0].split(', ')[1]
                Xmax = bbox[1].split(', ')[0]
                Ymax = bbox[1].split(', ')[1]

                width = int(Xmax) - int(Xmin)
                height = int(Ymax) - int(Ymin)

                aspect_ratios.append(width/height)

                box = ET.SubElement(image, 'box')
                box.attrib['top'] = Ymin
                box.attrib['left'] = Xmin
                box.attrib['width'] = str(width)
                box.attrib['height'] = str(height)

    tree = ET.ElementTree(ds)
    tree.write('annotations.xml')

