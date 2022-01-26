

# import libraries
from PIL import Image, ImageDraw
import PIL
import torch
import os
from torchvision import transforms
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as F
from torchvision.utils import save_image
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import os
import xml.dom.minidom
import glob
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# classes of the dataset
voc_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
# map classes to its corresponding number
label_map = {k: v+1 for v, k in enumerate(voc_labels)}
#Inverse mapping
rev_label_map = {v: k for k, v in label_map.items()}
#Colormap for bounding box
# number of classes (34)
CLASSES = 34
distinct_colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                   for i in range(CLASSES)]
label_color_map  = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}

image_path = "./test_1/"
annotation_path = "./test_1_labels/"

files_name = os.listdir(image_path)
for filename_ in files_name:
    filename, extension = os.path.splitext(filename_)
    img_path = image_path + filename + '.jpg'
    xml_path = annotation_path + filename + '.xml'
    print(img_path)
    img = cv2.imread(img_path)
    if img is None:
        pass
    dom = xml.dom.minidom.parse(xml_path)
    root = dom.documentElement
    objects = dom.getElementsByTagName("object")
    print(objects)
    i = 0
    xy = []
    for object in objects:
        bndbox = root.getElementsByTagName('bndbox')[i]
        xmin = bndbox.getElementsByTagName('xmin')[0]
        ymin = bndbox.getElementsByTagName('ymin')[0]
        xmax = bndbox.getElementsByTagName('xmax')[0]
        ymax = bndbox.getElementsByTagName('ymax')[0]
        xmin_data = xmin.childNodes[0].data
        ymin_data = ymin.childNodes[0].data
        xmax_data = xmax.childNodes[0].data
        ymax_data = ymax.childNodes[0].data
        print(object)
        print(xmin_data)
        print(ymin_data)
        xy.append([xmin_data, ymin_data, xmax_data, ymax_data])
        i = i + 1
        cv2.rectangle(img, (int(xmin_data), int(ymin_data)), (int(xmax_data), int(ymax_data)), (55, 255, 155), 5)

    X_min = int(min([i[0] for i in xy]))
    Y_min = int(min([i[1] for i in xy]))
    X_max = int(max([i[2] for i in xy]))
    Y_max = int(max([i[3] for i in xy]))

    cv2.imshow('img', img[Y_min - 5:Y_max + 5, X_min - 5:X_max + 5])
    cv2.waitKey(0)




    # flag = 0
    # flag = cv2.imwrite(
    #     "C:/Users/piyus/Documents/ObjectDetectionUdemy/PascalVOC-to-Images/data/Visualization/{}.jpg".format(filename),
    #     img)
    # if (flag):
    #     print(filename, "done")
print("all done ====================================")


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            try:
                value = (root.find('filename').text,
                         int(root.find('size')[0].text),
                         int(root.find('size')[1].text),
                         member[0].text,
                         int(member[4][0].text),
                         int(member[4][1].text),
                         int(member[4][2].text),
                         int(member[4][3].text)
                         )
                xml_list.append(value)
            except:
                pass
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


# apply the function to convert all XML files in images/ folder into labels.csv
labels_df = xml_to_csv('images/')
labels_df.to_csv(('labels.csv'), index=None)



