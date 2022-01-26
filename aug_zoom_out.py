


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


def parse_annot(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    boxes = list()
    labels = list()
    difficulties = list()

    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)



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


def draw_PIL_image(image, boxes, labels, new_h, new_w):
    '''
        Draw PIL image
        image: A PIL image
        labels: A tensor of dimensions (#objects,)
        boxes: A tensor of dimensions (#objects, 4)
    '''
    if not os.path.exists('./test_labels/%s.xml' % (image_id)):
        return

    in_file = open('./test_labels/%s.xml' % (image_id))

    out_file = open('contrast_labels/%s_zoom_out.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = new_w
    print('w_n', w)
    h = new_h
    print('h_n:', h)
    if type(image) != PIL.Image.Image:
        image = F.to_pil_image(image)
    new_image = image.copy()
    labels = labels.tolist()
    # print("Labels: ",labels)
    draw = ImageDraw.Draw(new_image)
    boxes = boxes.tolist()
    # print("boxes: ", boxes)
    for i, j in zip(labels, boxes):
        c = i
        b = tuple(j)
        # print(c)
        # print(b)
        bb = convert((w, h), b)
        # print(bb)

        out_file.write(str(c) + " " + " ".join([str(a) for a in bb]) + '\n')


    # for obj in root.iter('object'):
    #     cls = obj.find('name').text
    #     print(cls)
    #     if cls not in voc_labels:
    #         continue
    #     cls_id = voc_labels.index(cls)


    # for i in range(len(boxes)):
    #     cls = labels_new[i]
    #     print(cls)
    #     if cls not in voc_labels:
    #         continue
    #     cls_id = voc_labels.index(cls)
    #     b = tuple(boxes[i])
    #     bb = convert((w, h), b)
    #     out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    for i in range(len(boxes)):
        draw.rectangle(xy= boxes[i], outline= label_color_map[rev_label_map[labels[i]]])



def convert_annotation(image_id):

    if not os.path.exists('./test_labels/%s.xml' % (image_id)):
        return

    in_file = open('./test_labels/%s.xml' % (image_id))

    out_file = open('contrast_labels/%s_zoom_out.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    # print('w_n:', w)
    h = int(size.find('height').text)
    # print('h_n:', h)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        # print(cls)
        if cls not in voc_labels:
            continue
        cls_id = voc_labels.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        # print(b)
        bb = convert((w, h), b)
        # print(bb)
        # out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def zoom_out(image, boxes):
    '''
        Zoom out image (max scale = 4)
        image: A PIL image
        boxes: bounding boxes, a tensor of dimensions (#objects, 4)

        Out: new_image, new_boxes
    '''
    if type(image) == PIL.Image.Image:
        image = F.to_tensor(image)
    original_h = image.size(1)
    original_w = image.size(2)
    max_scale = 4
    scale = random.uniform(1, max_scale)
    new_h = int(scale * original_h)
    new_w = int(scale * original_w)

    print(new_w)
    print(new_h)


    # Create an image with the filler
    filler = [0.485, 0.456, 0.406]
    filler = torch.FloatTensor(filler)  # (3)
    new_image = torch.ones((3, new_h, new_w), dtype=torch.float) * filler.unsqueeze(1).unsqueeze(1)

    left = random.randint(0, new_w - original_w)
    right = left + original_w
    top = random.randint(0, new_h - original_h)
    bottom = top + original_h

    new_image[:, top:bottom, left:right] = image

    # Adjust bounding box
    new_boxes = boxes + torch.FloatTensor([left, top, left, top]).unsqueeze(0)
    # print("new_boxes: ", new_boxes)

    return new_image, new_boxes, new_h, new_w



for image in os.listdir('./test_images/'):
    #  It needs to be modified according to your picture . For example, the name of the picture is 123.456.jpg, There will be mistakes here . In general , If the picture format is fixed , If it's all jpg, It would be image_id=image[:-4] Just deal with it . All in all , There's a lot going on , Take matters into one's own hands , ha-ha ！

    image_id = image.split('.')[0]
    image = Image.open("./test_images/" + image_id +".jpg", mode="r")
    image = image.convert("RGB")
    objects = parse_annot("./test_labels/" + image_id + ".xml")
    boxes = torch.FloatTensor(objects['boxes'])
    labels = torch.LongTensor(objects['labels'])

    new_image, new_boxes, new_h, new_w = zoom_out(image, boxes)
    pil_image = transforms.ToPILImage(new_image)
    # print(pil_image)
    save_image(new_image, "./contrast/" +image_id + "_zoom_out.png")
    # pil_image.save("./contrast/" +image_id + "_zoom_out.png")
    draw_PIL_image(new_image, new_boxes, labels, new_h, new_w)
    # convert_annotation(image_id)





