import cv2
import numpy as np
from PIL import Image
import os
import numpy as np
import nibabel as nib
from matplotlib import pylab as plt
import random
import csv
import config

read_configuration = config._config

def find_inner_and_outer_boxes(boxes):
    biggest_box = max(boxes, key=lambda box: box[2] * box[3])  # Find the biggest box based on area
    inner_boxes = []
    outer_boxes = []

    for box in boxes:
        if box == biggest_box:
            continue

        if (
            box[0] >= biggest_box[0] and
            box[1] >= biggest_box[1] and
            (box[0] + box[2]) <= (biggest_box[0] + biggest_box[2]) and
            (box[1] + box[3]) <= (biggest_box[1] + biggest_box[3])
        ):
            inner_boxes.append(box)
        else:
            outer_boxes.append(box)

    return biggest_box, outer_boxes

def filter_boxes(boxes, min_area = 10):
    return [box for box in boxes if area(box) >= min_area]

def area(box):
    _, _, w, h = box
    return w * h

with open('bbox_coordinate.csv', 'w', encoding='UTF8') as f:
    header = ['img_slice_id', 'x', 'y', 'w', 'h', 'contours']
    writer = csv.writer(f)
    writer.writerow(header)

    prevPath = os.path.abspath('..')
    bfc_active = read_configuration['file_name']['bfc_active']
    filePath = os.path.join(prevPath, bfc_active)
    trainFilePath = os.path.join(filePath, 'train')
    trainFilePath = os.path.join(trainFilePath, 'gt')
    testFilePath = os.path.join(filePath, 'test')
    testFilePath = os.path.join(trainFilePath, 'gt')

    for root, dirs, files in os.walk(filePath):
        files.sort()
        for file in files:
            if file == '.DS_Store':
                continue
            path = os.path.join(root, file)
            img = nib.load(path)
            image_data = img.get_fdata()
            for slice_index in range(image_data.shape[2]):
                num = file.split('_')[2]
                name = "bb_" + num + "_" + str(slice_index)
                slice = image_data[:, :, slice_index]
                # gray = cv2.cvtColor(slice, cv2.COLOR_BGR2GRAY)
                ret, binary = cv2.threshold(slice, 0, 255, cv2.THRESH_BINARY)
                if binary.dtype != np.uint8:
                    binary = binary.astype(np.uint8)
                contours, hierarchy = cv2.findContours(binary,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                boxes = [cv2.boundingRect(contour) for contour in contours]
                boxes = filter_boxes(boxes, 10)

                if len(boxes) == 0:
                    coo_list = [name, 'NA', 'NA', 'NA', 'NA', str(len(boxes))]
                    writer.writerow(coo_list)
                    print('No box ' + name)

                else:
                    biggest_box, outer_boxes = find_inner_and_outer_boxes(boxes)

                    if not len(outer_boxes) == 0:
                        new_biggest_box, new_outer_boxes = find_inner_and_outer_boxes(outer_boxes)
                        new_outer_boxes.append(new_biggest_box)
                        new_outer_boxes.append(biggest_box)
                        print('write ' + name)
                        for box in new_outer_boxes:
                            x, y, w, h = box
                            coo_list = [name, x, y, w, h, str(len(new_outer_boxes))]
                            writer.writerow(coo_list)
                    else:
                        outer_boxes.append(biggest_box)
                        print('write ' + name)
                        for box in outer_boxes:
                            x, y, w, h = box
                            coo_list = [name, x, y, w, h, str(len(outer_boxes))]
                            writer.writerow(coo_list)
    for root, dirs, files in os.walk(testFilePath):
        files.sort()
        for file in files:
            if file == '.DS_Store':
                continue
            path = os.path.join(root, file)
            img = nib.load(path)
            image_data = img.get_fdata()
            for slice_index in range(image_data.shape[2]):
                num = file.split('_')[2]
                name = "bb_" + num + "_" + str(slice_index)
                slice = image_data[:, :, slice_index]
                # gray = cv2.cvtColor(slice, cv2.COLOR_BGR2GRAY)
                ret, binary = cv2.threshold(slice, 0, 255, cv2.THRESH_BINARY)
                if binary.dtype != np.uint8:
                    binary = binary.astype(np.uint8)
                contours, hierarchy = cv2.findContours(binary,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                boxes = [cv2.boundingRect(contour) for contour in contours]
                boxes = filter_boxes(boxes, 10)

                if len(boxes) == 0:
                    coo_list = [name, 'NA', 'NA', 'NA', 'NA', str(len(boxes))]
                    writer.writerow(coo_list)
                    print('No box ' + name)

                else:
                    biggest_box, outer_boxes = find_inner_and_outer_boxes(boxes)

                    if not len(outer_boxes) == 0:
                        new_biggest_box, new_outer_boxes = find_inner_and_outer_boxes(outer_boxes)
                        new_outer_boxes.append(new_biggest_box)
                        new_outer_boxes.append(biggest_box)
                        print('write ' + name)
                        for box in new_outer_boxes:
                            x, y, w, h = box
                            coo_list = [name, x, y, w, h, str(len(new_outer_boxes))]
                            writer.writerow(coo_list)
                    else:
                        outer_boxes.append(biggest_box)
                        print('write ' + name)
                        for box in outer_boxes:
                            x, y, w, h = box
                            coo_list = [name, x, y, w, h, str(len(outer_boxes))]
                            writer.writerow(coo_list)
