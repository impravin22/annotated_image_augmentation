import albumentations
import albumentations as A
import cv2
import os
import xml.etree.ElementTree as ET
from os import getcwd
import json
from collections import defaultdict
import matplotlib.pyplot as plt

transform_type = 'affine' # vFlip, hFlip, noise, paffine, affine, scaleShift, blur


def out_of_image_bbox(func):
    def ignore_error():
        try:
            return ignore_error()
        except ValueError as v:
            print('Value error! skipping image')
    return ignore_error()

dataset_path = './dataset/train/'
labels_path = './dataset/labels/'
json_labels_path = './dataset/train_json/'

sets = ['train']

class_labels = ["class 1", "class 2"]

# transform = A.Compose([A.HorizontalFlip(p=0.5),
#                       A.RandomBrightnessContrast(p=0.2)], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

transform_hFlip = A.Compose([A.HorizontalFlip(always_apply=True, p=1.0)],
                          bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
transform_vFlip = A.Compose([A.VerticalFlip(always_apply=True, p=1.0)],
                          bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
transform_affine = A.Compose([A.Affine(always_apply=True, p=1.0)],
                          bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
transform_paffine = A.Compose([A.PiecewiseAffine(always_apply=True, p=1.0)],
                          bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
transform_scaleShift = A.Compose([A.ShiftScaleRotate(always_apply=True, p=1.0)],
                          bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
transform_noise = A.Compose([A.GaussNoise(var_limit=(640.0, 690.0), always_apply=True, p=1.0)],
                          bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
transform_blur = A.Compose([A.GaussianBlur(blur_limit=(21, 23), always_apply=True, p=1.0)],
                          bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))


# transform_affine = A.Affine(always_apply=True, p=1)

data = json.load(open('./dataset/train_data.json'))



def get_bbox_from_json(annotation_key, cat_id, image_key, f_name):
    image_list = []
    bbox_list = []
    class_list = []
    for i in range(len(data[annotation_key])):

        id = data[annotation_key][i]['image_id']
        data_img = data[image_key]
        img_name_list = []
        for img_data in data_img:
            if img_data['id'] == id:
                img_name_list.append(img_data)
        image_id = f'{img_name_list[0][f_name]}'
        image_list.append(image_id)
        image_id = image_id.split('.png')[0]
        category_id = f'{int(data[annotation_key][i][cat_id])}'
        bbox = data[annotation_key][i]['bbox']
        bbox_list.append(bbox)
        class_label = data[annotation_key][i]['id']
        if class_label == 1.0:
            class_list.append('class 2')
        elif class_label == 0.0:
            class_list.append('class 1')
    return image_list, bbox_list, class_list

annotation_key = 'annotations'
cat_id = 'id'
image_key = 'images'
f_name = 'file_name'

img_ls, bbox_ls, class_ls = get_bbox_from_json(annotation_key, cat_id, image_key, f_name)
unique_img_index = [img_ls.index(i) for i in set(img_ls)]

img_indices = defaultdict(list)

for index, item in enumerate(img_ls):
    img_indices[item].append(index)
img_indices_dict = dict(img_indices)

# img_name = list(img_indices_dict.keys())[0]
# label_list = list(img_indices_dict.values())[0]
#
# img_full_path = dataset_path + img_name
# label_bbox = [bbox_ls[label] for label in label_list]
# class_labels = [class_ls[label] for label in label_list]
# image = cv2.imread(img_full_path)
# # for i in label_bbox:
# #     cv2.rectangle(image, (int(i[0]), int(i[1])), (int(i[2] + i[0]), int(i[3] + i[1])), (0, 255, 255), 2)
#

for i in range(len(list(img_indices_dict))):
    image_name = list(img_indices_dict.keys())[i]
    label_list = list(img_indices_dict.values())[i]
    bbox_list = [bbox_ls[label] for label in label_list]
    class_list = [class_ls[label] for label in label_list]

    data_ann_list = [data[annotation_key][label]['bbox'] for label in label_list]
    data_img = data['images'][i]['file_name']
    data_img_width = data['images'][i]['width']
    data_img_height = data['images'][i]['height']
    data_class_list = [data[annotation_key][label]['id'] for label in label_list]
    # for i in bbox_list:
    #     i[2] = i[0] + i[2]
    #     i[3] = i[1] + i[3]
    img_full_path = dataset_path + image_name
    print(f'label_list is: {label_list}')
    data_ann_image_id = [data[annotation_key][i]['image_id'] for i in range(len(data[annotation_key]))]
    ls = [label_list.index(i) for i in label_list]

    print(f'list ls: {ls}')
    image = cv2.imread(img_full_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # transformed = transform_hFlip(image=image, bboxes=bbox_list, class_labels=class_list)

    try:
        if transform_type == 'affine':
            transformed = transform_affine(image=image, bboxes=bbox_list, class_labels=class_list)
        elif transform_type == 'hFlip':
            transformed = transform_hFlip(image=image, bboxes=bbox_list, class_labels=class_list)
        elif transform_type == 'vFlip':
            transformed = transform_vFlip(image=image, bboxes=bbox_list, class_labels=class_list)
        elif transform_type == 'paffine':
            transformed = transform_paffine(image=image, bboxes=bbox_list, class_labels=class_list)
        elif transform_type == 'scaleShift':
            transformed = transform_scaleShift(image=image, bboxes=bbox_list, class_labels=class_list)
        elif transform_type == 'noise':
            transformed = transform_noise(image=image, bboxes=bbox_list, class_labels=class_list)
        elif transform_type == 'blur':
            transformed = transform_blur(image=image, bboxes=bbox_list, class_labels=class_list)

        transformed_image = transformed['image']
        transformed_bbox = transformed['bboxes']
        transformed_bbox = [list(i) for i in transformed_bbox]
        transformed_class_labels = transformed['class_labels']

        augmented_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
        image_name = image_name.split('.png')[0]
        new_image_name = str(image_name) + '_' + str(transform_type) + '.png'
        data['images'][i]['file_name'] = new_image_name
        data['images'][i]['width'] = augmented_image.shape[1]
        data['images'][i]['height'] = augmented_image.shape[0]
        data_ann_image_id = [data[annotation_key][i]['image_id'] for i in range(len(data[annotation_key]))]
        for label in label_list:
            data[annotation_key][label]['bbox'] = transformed_bbox[label_list.index(label)]
            data[annotation_key][label]['id'] = class_labels.index(transformed_class_labels[label_list.index(label)])
        transformed_ann_list = [data[annotation_key][label]['bbox'] for label in label_list]
        transformed_class_list = [data[annotation_key][label]['id'] for label in label_list]
        cv2.imwrite('./augmented_dataset/' + str(transform_type) + '/' + str(new_image_name), augmented_image)




        # for i in range(len(bbox_list)):
        #     cv2.rectangle(image, (int(bbox_list[i][0]), int(bbox_list[i][1])), (int(bbox_list[i][0]) + int(bbox_list[i][2]), int(bbox_list[i][1]) + int(bbox_list[i][3])), (255, 255, 0), 2)
        #     # cv2.rectangle(image, (int(bbox_list[1][0]), int(bbox_list[1][1])), (int(bbox_list[1][0]) + int(bbox_list[1][2]), int(bbox_list[1][1]) + int(bbox_list[1][3])), (255, 255, 0), 2)
        # for i in range(len(transformed_bbox)):
        #     cv2.rectangle(transformed_image, (int(transformed_bbox[i][0]), int(transformed_bbox[i][1])), (int(transformed_bbox[i][0]) + int(transformed_bbox[i][2]),
        #                                                                                               int(transformed_bbox[i][1]) + int(transformed_bbox[i][3])), (0, 255, 0), 2)

        # f, ax = plt.subplots(1, 2)
        # ax[1].imshow(transformed_image)
        # ax[0].imshow(image)
        # # plt.imshow()
        # plt.pause(2)
        # plt.close(f)
    except ValueError as v:
        print(f'skipping image because of {v}')
    except IndexError as i:
        print(f'skipping image because of {i}')


    print(image_name)
    print(bbox_list)
    print(transformed_ann_list)

with open('./augmented_dataset/train_data_' + str(transform_type) + '.json', 'w') as json_to_write:
    # with open('./eye_key_point_dataset/test/test_val.json', 'w') as json_to_write:
    #     json_to_write.write(data)
    json.dump(data, json_to_write)
    print('written successfully')

# cv2.imshow('image', image)
# cv2.waitKey(0)
# print(img_ls)
