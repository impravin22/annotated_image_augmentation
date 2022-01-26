
# import libraries
from PIL import Image, ImageDraw
import PIL
import torch
import os
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as F
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt

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


def parse_annot(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    boxes = list()
    labels = list()
    difficulties = list()

    for object in root.iter("object"):
        difficult = int(object.find("difficult").text == "1")
        label = object.find("name").text.upper().strip()
        if label not in label_map:
            print("{0} not in label map.".format(label))
            assert label in label_map

        bbox = object.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[label])
        difficulties.append(difficult)

    return {"boxes": boxes, "labels": labels, "difficulties": difficulties}

def draw_PIL_image(image, boxes, labels):
    '''
        Draw PIL image
        image: A PIL image
        labels: A tensor of dimensions (#objects,)
        boxes: A tensor of dimensions (#objects, 4)
    '''
    if type(image) != PIL.Image.Image:
        image = F.to_pil_image(image)
    new_image = image.copy()
    labels = labels.tolist()
    print("Labels: ",labels)
    draw = ImageDraw.Draw(new_image)
    boxes = boxes.tolist()
    print("boxes: ", boxes)
    for i in range(len(boxes)):
        draw.rectangle(xy= boxes[i], outline= label_color_map[rev_label_map[labels[i]]])

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(image_id):

    if not os.path.exists('./carin_LP_labels/%s.xml' % (image_id)):
        return

    in_file = open('./carin_LP_labels/%s.xml' % (image_id))

    out_file = open('./aug_carin_LP_labels/%s_saturation.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in voc_labels:
            continue
        cls_id = voc_labels.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        print(b)
        bb = convert((w, h), b)
        # print(bb)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

def Adjust_saturation(image):
    # adjust saturation by 2
    return F.adjust_saturation(image, 2)



for image in os.listdir('./carin_LP/'):
    #  It needs to be modified according to your picture . For example, the name of the picture is 123.456.jpg, There will be mistakes here . In general , If the picture format is fixed , If it's all jpg, It would be image_id=image[:-4] Just deal with it . All in all , There's a lot going on , Take matters into one's own hands , ha-ha ！

    image_id = image.split('.jpg')[0]
    image = Image.open("./carin_LP/" + image_id +".jpg", mode="r")
    image = image.convert("RGB")
    objects = parse_annot("./carin_LP_labels/" + image_id + ".xml")
    boxes = torch.FloatTensor(objects['boxes'])
    labels = torch.LongTensor(objects['labels'])

    new_image = Adjust_saturation(image)
    new_image.save("./aug_carin_LP/" +image_id + "_saturation.png")
    draw_PIL_image(new_image, boxes, labels)
    convert_annotation(image_id)