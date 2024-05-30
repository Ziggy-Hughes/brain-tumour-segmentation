import os
import config
import numpy as np
import pandas as pd
import nibabel as nib
from collections import defaultdict
import math
import cv2

read_configuration = config._config

class Preprocess(object):

    def read_config(self):
        # remove top and bottom slices
        self.remove_slice_num = read_configuration['param_setting']['remove_slice_num']
        self.max_img_height = 216
        self.max_img_width = 240

    def get_folder_path(self, dataset):
        curPath = os.path.abspath('.')
        prevPath = os.path.abspath('..')

        bbox_name = read_configuration['file_name']['csv_file_name']
        self.bbox_coord_fpath = os.path.join(curPath, bbox_name)

        bfc_active = read_configuration['file_name']['bfc_active']
        filePath = os.path.join(prevPath, bfc_active)

        if dataset == 'train':
            filePath = os.path.join(filePath, 'train')
            self.img_folder_path = os.path.join(filePath, 'img')
            self.gt_folder_path = os.path.join(filePath, 'gt')
        elif dataset == 'test':
            filePath = os.path.join(filePath, 'test')
            self.img_folder_path = os.path.join(filePath, 'img')
            self.gt_folder_path = os.path.join(filePath, 'gt')
        else:
            filePath = os.path.join(filePath, 'train')
            self.img_folder_path = os.path.join(filePath, 'img')
            self.gt_folder_path = os.path.join(filePath, 'gt')

    def get_file_list(self):
        self.img_file_list = os.listdir(self.img_folder_path)
        self.gt_file_list = os.listdir(self.gt_folder_path)
        self.gt_id_list = []
        for gt_fname in self.gt_file_list:
            if gt_fname != '.DS_Store':
                gt_id = gt_fname.split('_')[2]
                self.gt_id_list.append(gt_id)

    def read_bbox_coord(self):
        self.bbox_coord_df = pd.read_csv(self.bbox_coord_fpath)
        self.bbox_img_slice_id_list = self.bbox_coord_df['img_slice_id'].values.tolist()

    def match_img_and_gt(self, img_id):
        if img_id not in self.gt_id_list:
            print("We cannot find the corresponding ground truth of img %s,"
                  "Continue to Check the Next Image File!" % (img_id))
            return False
        else:
            return True

    def read_nii_data(self, img_id, img_nii_name):
        gt_fname = "BraTS20_Training_" + img_id + "_seg.nii.gz"
        img_fpath = os.path.join(self.img_folder_path, img_nii_name)
        gt_fpath = os.path.join(self.gt_folder_path, gt_fname)
        nib_img = nib.load(img_fpath)
        nib_gt = nib.load(gt_fpath)
        img_array_3d = nib_img.get_fdata()
        gt_array_3d = nib_gt.get_fdata()
        self.img_array_3d = img_array_3d
        self.gt_array_3d = gt_array_3d

        return img_array_3d, gt_array_3d

    def img_transform(self, input_img, is_img):
        # 216x240x155 -> 155x216x240
        output_img = input_img.transpose(2, 0, 1)
        # 155x216x240 -> 155x1x216x240
        output_img = np.expand_dims(output_img, axis = 1)
        if is_img:
            # 155x1x216x240 -> 155x3x216x240
            output_img = np.repeat(output_img, 3, 1)
        return output_img

    def trim_box_coordinates(self, box_coordinates):

        trimmed_box_coordinates = []
        for box in box_coordinates:
            x, y, width, height = box
            trimmed_box = (x, y - 12, width, height)
            trimmed_box_coordinates.append(trimmed_box)

        return trimmed_box_coordinates

    def create_mask(self, bbox_coords, img_height, img_width):
        mask = np.zeros((img_height, img_width))
        for box in bbox_coords:
            if any(math.isnan(value) for box in bbox_coords for value in box):
                mask = np.zeros((img_height, img_width))
            else:
                x, y, w, h = box
                mask[y:y+h, x:x+w] = 1
        return mask

    def get_bbox_area(self, img_data, bbox_coords):
        bbox_area_array_2d_list = []

        for box in bbox_coords:
            if any(math.isnan(value) for box in bbox_coords for value in box):
                return []
            else:
                x, y, w, h = box
                bbox_area = img_data[y:y+h, x:x+w]
                bbox_area_array_2d_list.append(bbox_area)
        return bbox_area_array_2d_list

    def get_bbox_label(self, img_id, slice_id, img_height, img_width, img_2d):
        x_bbox_coord_list = [int(x) if not math.isnan(x) else float('nan') for x in self.bbox_coord_df['x']]
        y_bbox_coord_list = [int(x) if not math.isnan(x) else float('nan') for x in self.bbox_coord_df['y']]
        w_bbox_coord_list = [int(x) if not math.isnan(x) else float('nan') for x in self.bbox_coord_df['w']]
        h_bbox_coord_list = [int(x) if not math.isnan(x) else float('nan') for x in self.bbox_coord_df['h']]

        all_bbox_coord_list = []
        for i in range(len(x_bbox_coord_list)):
            all_bbox_coord_list.append((x_bbox_coord_list[i], y_bbox_coord_list[i], w_bbox_coord_list[i], h_bbox_coord_list[i]))

        all_bbox_coord_dict = defaultdict(list)
        for i, value in enumerate(self.bbox_img_slice_id_list):
            all_bbox_coord_dict[value].append(all_bbox_coord_list[i])

        img_slice_id = 'bb_' + img_id + '_' + str(slice_id)
        if img_slice_id in all_bbox_coord_dict.keys():
            temp_slice_bbox_coord = all_bbox_coord_dict[img_slice_id]

        slice_bbox_coord = self.trim_box_coordinates(temp_slice_bbox_coord)

        print("img_slice_id:", img_slice_id)
        mask_label = self.create_mask(slice_bbox_coord, 216, 240)
        bbox_area = self.get_bbox_area(img_2d, slice_bbox_coord)

        return mask_label, bbox_area, slice_bbox_coord

    def delete_nan(self, bbox_coords):
        for box in bbox_coords:
            if any(math.isnan(value) for box in bbox_coords for value in box):
                return []
            else:
                return bbox_coords

    def adjust_mask(self, pixel_mask):
        num_labels, labeled_array = cv2.connectedComponents(pixel_mask.astype(np.uint8), connectivity=8)
        unique_labels, label_counts = np.unique(labeled_array, return_counts=True)

        largest_label = unique_labels[np.argmax(label_counts[1:]) + 1]
        largest_component_array = (np.where(labeled_array == largest_label, 1, 0)).astype(np.float32)

        padding = 10
        padded_height = largest_component_array.shape[0] + 2 * padding
        padded_width = largest_component_array.shape[1] + 2 * padding
        padded_array = np.zeros((padded_height, padded_width), dtype=largest_component_array.dtype)
        padded_array[padding:padded_height-padding, padding:padded_width-padding] = largest_component_array

        im_floodfill = padded_array.copy()

        h, w = padded_array.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)

        # Will always be (0, 0) due to padding?
        seedPoint = None
        for i in range(im_floodfill.shape[0]):
            for j in range(im_floodfill.shape[1]):
                if im_floodfill[i][j] == 0:
                    seedPoint = (j, i)
                    break
            if seedPoint is not None:
                break
        cv2.floodFill(im_floodfill, mask, seedPoint, 255)
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        im_out = cv2.bitwise_or(padded_array, im_floodfill_inv)
        im_out[~np.isnan(im_out)] = 0
        im_out[np.isnan(im_out)] = 1

        im_out_height = im_out.shape[0] - 2 * padding
        im_out_width = im_out.shape[1] - 2 * padding

        result = im_out[padding:padding+im_out_height, padding:padding+im_out_width]

        label_image = np.uint8(result) * 255

        contours, _ = cv2.findContours(label_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        hull = cv2.convexHull(cnt)
        convexHull_array = np.zeros(np.shape(label_image), np.uint8)
        cv2.fillPoly(convexHull_array, [hull], 1)

        convexHull_array = convexHull_array.astype(np.float32)

        return convexHull_array

    def kmeans(self, box_area_list):
        kmeans_area_list = []

        def preprocess_image(image):

            normalized_image = image.astype(np.float32) / 255.0
            resized_image = normalized_image
            blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)

            return blurred_image

        def kmeans_segmentation(image, num_clusters):
            pixel_values = image.reshape(-1, 3).astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
            _, labels, centers = cv2.kmeans(pixel_values, num_clusters, None, criteria, 30, cv2.KMEANS_RANDOM_CENTERS)
            segmented_image = centers[labels.flatten()].reshape(image.shape)

            return segmented_image

        for i in range(len(box_area_list)):
            if not box_area_list[i].any():
                box_area_value = np.zeros(np.shape(box_area_list[i]), dtype = np.float32)
                kmeans_area_list.append(box_area_value)
                continue

            image = np.stack((box_area_list[i],) * 3, axis=-1)

            processed_image = preprocess_image(image)

            height = np.shape(processed_image)[0]
            width = np.shape(processed_image)[1]

            pixel_mask = np.zeros((height, width), dtype = np.float32)

            if len(image) == 1:
                result = pixel_mask
                kmeans_area_list.append(result)
            elif len(image) == 2:
                num_clusters = 2
                segmented_image = kmeans_segmentation(processed_image, num_clusters)
                segmented_image_squeeze = np.mean(segmented_image, axis=2)
                segmented_image_around = np.around(segmented_image_squeeze, 2)
                segmented_image_unique = np.sort(np.unique(segmented_image_around))

                if len(segmented_image_unique) == 1:
                    norm_tumour_value = segmented_image_unique[0]
                else:
                    norm_tumour_value = segmented_image_unique[1]

                for i in range(len(segmented_image_around)):
                    for j in range(len(segmented_image_around[i])):
                        temp_pixel_value = segmented_image_around[i][j]
                        if temp_pixel_value == norm_tumour_value:
                            pixel_mask[i][j] = 1

                pixel_unique = np.unique(pixel_mask)
                if list(pixel_unique) == [0.] or list(pixel_unique) == [1.]:
                    result = pixel_mask
                else:
                    result = self.adjust_mask(pixel_mask)

                kmeans_area_list.append(result)
            else:
                num_clusters = 3
                segmented_image = kmeans_segmentation(processed_image, num_clusters)
                segmented_image_squeeze = np.mean(segmented_image, axis=2)
                segmented_image_around = np.around(segmented_image_squeeze, 10)
                segmented_image_unique = np.sort(np.unique(segmented_image_around))

                if len(segmented_image_unique) == 1:
                    norm_tumour_value = segmented_image_unique[0]
                else:
                    norm_tumour_value = segmented_image_unique[1]

                for i in range(len(segmented_image_around)):
                    for j in range(len(segmented_image_around[i])):
                        temp_pixel_value = segmented_image_around[i][j]
                        if temp_pixel_value == norm_tumour_value:
                            pixel_mask[i][j] = 1

                pixel_unique = np.unique(pixel_mask)
                if list(pixel_unique) == [0.] or list(pixel_unique) == [1.]:
                    result = pixel_mask
                else:
                    result = self.adjust_mask(pixel_mask)

                kmeans_area_list.append(result)

        self.k_list = kmeans_area_list

        return kmeans_area_list

    def threshold(self, box_area_list):
        threshold_1 = 190 #white
        threshold_2 = 36 #black
        threshold_area_list = []

        for i in range(len(box_area_list)):
            if not box_area_list[i].any():
                box_area_value = np.zeros(np.shape(box_area_list[i]), dtype = np.float32)
                threshold_area_list.append(box_area_value)
                continue

            data_2d = box_area_list[i]
            height = np.shape(data_2d)[0]
            width = np.shape(data_2d)[1]

            pixel_mask = np.zeros((height, width))

            for i in range(height):
                for j in range(width):
                    pixel_value = data_2d[i][j]
                    if pixel_value >= threshold_1:
                        pixel_mask[i][j] = 2
                    if threshold_2 <= pixel_value < threshold_1:
                        pixel_mask[i][j] = 1

            pixel_mask[pixel_mask == 2] = 0

            pixel_unique = np.unique(pixel_mask)

            if list(pixel_unique) == [0.] or list(pixel_unique) == [1.]:
                result = pixel_mask
            else:
                result = self.adjust_mask(pixel_mask)
            threshold_area_list.append(result)

        self.threshold_list = threshold_area_list

        return threshold_area_list

    def jigsaw(self, jigsaw_method, input_area_list, box_coord, image_shape):
        if jigsaw_method == 'kmeans':
            box_area_list = self.kmeans(input_area_list)
        elif jigsaw_method == 'threshold':
            box_area_list = self.threshold(input_area_list)

        label = np.zeros(image_shape)

        if [] in box_coord:
            return label
        else:
            for i, coord in enumerate(box_coord):
                x, y, w, h = coord
                box_area = box_area_list[i]

                x1, y1 = x, y
                x2, y2 = x + w, y
                x3, y3 = x + w, y + h
                x4, y4 = x, y + h

                x1, y1 = max(0, min(x1, image_shape[1] - 1)), max(0, min(y1, image_shape[0] - 1))
                x2, y2 = max(0, min(x2, image_shape[1] - 1)), max(0, min(y2, image_shape[0] - 1))
                x3, y3 = max(0, min(x3, image_shape[1] - 1)), max(0, min(y3, image_shape[0] - 1))
                x4, y4 = max(0, min(x4, image_shape[1] - 1)), max(0, min(y4, image_shape[0] - 1))

                label[y1:y3, x1:x3] = np.maximum(label[y1:y3, x1:x3], box_area)

            return label

    def concat_slices(self, dataset):
        count = 0

        self.img_file_list.sort()

        for img_nii_name in self.img_file_list:

            print("img_name:", img_nii_name)
            if img_nii_name != '.DS_Store':
                img_id = img_nii_name.split('_')[2]
                img_type = img_nii_name.split('_')[3][:-7]

                # only use t2 to reduce memory
                if img_type != 't2':
                    continue

                is_matched = self.match_img_and_gt(img_id)
                if not is_matched:
                    continue

                (img_array_3d, gt_array_3d) = self.read_nii_data(img_id, img_nii_name)

                # 240x240x155 -> 216x240x155
                img_array_3d_trim = img_array_3d[12:-12, :, :]
                gt_array_3d_trim = gt_array_3d[12:-12, :, :]

                img_array_4d_trim_trans = self.img_transform(img_array_3d_trim, True)
                gt_array_4d_trim_trans = self.img_transform(gt_array_3d_trim, False)

                img_height = len(img_array_3d_trim)
                img_width = len(img_array_3d_trim[0])
                total_num_slice = len(img_array_3d_trim[0][0])

                print("img_id = %s, total_num_slice = %d" % (img_id, total_num_slice))
                start_slice_id = 0 + self.remove_slice_num
                end_slice_id = total_num_slice - self.remove_slice_num
                count = count+1

                for slice_id in range(start_slice_id, end_slice_id):
                    img_array_2d = img_array_4d_trim_trans[slice_id,:,:,:]
                    gt_array_2d = gt_array_4d_trim_trans[slice_id,:,:,:]

                    gt_array_2d[gt_array_2d != 0] = 1

                    img_array_2d_for_box_area = img_array_3d_trim[:, :, slice_id]

                    mask_label, bbox_area, bbox_coord = self.get_bbox_label(img_id, slice_id, img_height, img_width, img_array_2d_for_box_area)

                    mask_label_3d = np.expand_dims(mask_label, axis=0)

                    slice_bbox_coord = self.delete_nan(bbox_coord)

                    kmeans_2d_array = self.jigsaw('kmeans', bbox_area, slice_bbox_coord, (img_height, img_width))
                    kmeans_3d_array = np.expand_dims(kmeans_2d_array, axis=0)

                    threshold_2d_array = self.jigsaw('threshold', bbox_area, slice_bbox_coord, (img_height, img_width))
                    threshold_3d_array = np.expand_dims(threshold_2d_array, axis=0)

                    data = self.reduce_data([img_array_2d, gt_array_2d, mask_label_3d, kmeans_3d_array, threshold_3d_array])

                    self.save_proc_data(dataset, img_id, img_type, slice_id, data)

                print('****** file count =', count, '****** ')

    def reduce_data(self, data):
        img_array_2d_reduced = np.zeros((3, 72, 80))
        for i in range(1):
            img_array_2d_reduced[i, :, :] = cv2.resize(data[0][i], dsize=(80, 72), interpolation=cv2.INTER_NEAREST)
        data[0] = img_array_2d_reduced

        gt_array_2d_reduced = np.zeros((1, 72, 80))
        gt_array_2d_reduced[0, :, :] = cv2.resize(data[1][0], dsize=(80, 72), interpolation=cv2.INTER_NEAREST)
        data[1] = gt_array_2d_reduced

        mask_label_3d_reduced = np.zeros((1, 72, 80))
        mask_label_3d_reduced[0, :, :] = cv2.resize(data[2][0], dsize=(80, 72), interpolation=cv2.INTER_NEAREST)
        data[2] = mask_label_3d_reduced

        kmeans_3d_array_reduced = np.zeros((1, 72, 80))
        kmeans_3d_array_reduced[0, :, :] = cv2.resize(data[3][0], dsize=(80, 72), interpolation=cv2.INTER_NEAREST)
        data[3] = kmeans_3d_array_reduced

        threshold_3d_array_reduced = np.zeros((1, 72, 80))
        threshold_3d_array_reduced[0, :, :] = cv2.resize(data[4][0], dsize=(80, 72), interpolation=cv2.INTER_NEAREST)
        data[4] = threshold_3d_array_reduced

        return data

    def save_proc_data(self, dataset, img_id, img_type, slice_id, data):
        parent_folder_path = os.path.abspath('..')
        saved_folder_name = 'ProcessedData'
        saved_folder_path = os.path.join(parent_folder_path, saved_folder_name)
        saved_folder_path = os.path.join(saved_folder_path, dataset)
        if not os.path.exists(saved_folder_path):
            os.makedirs(saved_folder_path)

        saved_data_name = img_id + '_' + img_type + '_' + str(slice_id)
        saved_data_path = os.path.join(saved_folder_path, saved_data_name)
        np.save(saved_data_path, data)

    def process(self):
        self.read_config()
        for dataset in ['train', 'test']:
            self.get_folder_path(dataset)
            self.get_file_list()
            self.read_bbox_coord()
            self.concat_slices(dataset)



proc = Preprocess()
proc.process()
