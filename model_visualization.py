import torch
import os
import config
import numpy as np
import evaluation
import set_logger
import logging
from matplotlib import pylab as plt
import matplotlib.colors as mcolors
read_configuration = config._config

try:
    import _pickle as pickle
except ImportError:
    import pickle

setlogger = set_logger.Setlogger()
setlogger.create_logger()
logger = logging.getLogger('root')


class AllModelVisualization(object):
    def __init__(self):
        print('****** Initialize Visualization Class! ******')
        self.load_train_test_pickel()
        self.set_parameters()
        self.load_segmentation_model()

    def set_parameters(self):
        print('****** Set Parameters! ******')
        self.plot_train_or_test = 'test'
        self.bg_pixel_value = 0
        self.tumour_pixel_value= 50

        self.bg_label_value = 0
        self.tumour_label_value = 1


        self.transparency = 0.5

        self.fontsize = 18


    def load_train_test_pickel(self):
        logger.info('****** Load Train Test Pickel! ******')

        prevPath = os.path.abspath('..')
        folder_path = os.path.join(prevPath, 'ProcessedData')
        train_folder_path = os.path.join(folder_path, 'train')
        test_folder_path = os.path.join(folder_path, 'test')

        train_file_list = os.listdir(train_folder_path)
        test_file_list = os.listdir(test_folder_path)

        self.picked_tuple_train_idx = 95 # change for best display
        self.picked_tuple_test_idx = 3080 # change for best display

        if self.picked_tuple_train_idx >= len(train_file_list):
            self.picked_tuple_train_idx = self.picked_tuple_train_idx % len(train_file_list)

        if self.picked_tuple_test_idx >= len(test_file_list):
            self.picked_tuple_test_idx = self.picked_tuple_test_idx % len(test_file_list)

        train_fpath = os.path.join(train_folder_path, train_file_list[self.picked_tuple_train_idx])
        test_fpath = os.path.join(test_folder_path, test_file_list[self.picked_tuple_test_idx])

        self.train_tensor = np.load(train_fpath, allow_pickle=True)
        self.test_tensor = np.load(test_fpath, allow_pickle=True)


    def load_segmentation_model(self):
        print('****** Load Model! ******')
        prev_folder_path = os.path.abspath('..')

        full_model_name = read_configuration['full_model_name']
        U_Net_model_path = prev_folder_path +'/saved_model/'+ full_model_name

        ws_model_name = read_configuration['ws_model_name']
        ws_U_Net_model_path = prev_folder_path +'/saved_model/'+ ws_model_name

        kmeans_model_name = read_configuration['kmeans_model_name']
        kmeans_U_Net_model_path = prev_folder_path +'/saved_model/'+ kmeans_model_name


        threshold_model_name = read_configuration['threshold_model_name']
        threshold_U_Net_model_path = prev_folder_path +'/saved_model/'+ threshold_model_name



        self.U_Net_model = torch.load(U_Net_model_path, map_location='cpu')
        self.U_Net_model.eval()


        self.ws_U_Net_model = torch.load(ws_U_Net_model_path, map_location='cpu')
        self.ws_U_Net_model.eval()


        self.kmeans_U_Net_model = torch.load(kmeans_U_Net_model_path, map_location='cpu')
        self.kmeans_U_Net_model.eval()

        self.threshold_U_Net_model = torch.load(threshold_U_Net_model_path, map_location='cpu')
        self.threshold_U_Net_model.eval()


    def process_img_gt_tensor(self):
        print('***** Process Image groundtruth Tensor! *****')

        def prepare_seg_results(seg_result_4d_tensor):
            seg_result_3d_tensor = seg_result_4d_tensor.squeeze(0)
            print('seg_result_3d_tensor:', seg_result_3d_tensor.size())
            seg_result_max_2d_tensor = torch.max(seg_result_3d_tensor, 0)[1]
            seg_result_2d_array = seg_result_max_2d_tensor.detach().cpu().numpy() #.detach(): detach from computational graph, so that no gradients
                                                                                  #can be calculated anymore
                                                                                  #.cpu(): switch gpu to cpu device
            return seg_result_2d_array


        if self.plot_train_or_test == 'train':
            img_3d_tensor = self.train_tensor[0]
            gt_3d_tensor = self.train_tensor[1]
            ws_3d_tensor = self.train_tensor[2]
            kmeans_3d_tensor = self.train_tensor[3]
            threshold_3d_tensor = self.train_tensor[4]


        elif self.plot_train_or_test == 'test':
            img_3d_tensor = self.test_tensor[0]
            gt_3d_tensor = self.test_tensor[1]
            ws_3d_tensor = self.test_tensor[2]
            kmeans_3d_tensor = self.test_tensor[3]
            threshold_3d_tensor = self.test_tensor[4]

        else:
            img_3d_tensor = self.test_tensor[0]
            gt_3d_tensor = self.test_tensor[1]
            ws_3d_tensor = self.test_tensor[2]
            kmeans_3d_tensor = self.test_tensor[3]
            threshold_3d_tensor = self.test_tensor[4]


        self.sliced_img_2d_array = img_3d_tensor[0]
        self.sliced_gt_2d_array = gt_3d_tensor[0]
        self.sliced_ws_2d_array = ws_3d_tensor[0]
        self.sliced_kmeans_2d_array = kmeans_3d_tensor[0]
        self.sliced_threshold_2d_array = threshold_3d_tensor[0]


        img_4d_tensor = np.expand_dims(img_3d_tensor, axis=0)

        img_4d_tensor_torch = torch.tensor(img_4d_tensor, dtype=torch.float32)

        U_Net_seg_result_4d_tensor = self.U_Net_model(img_4d_tensor_torch)
        self.U_Net_seg_result_2d_array = prepare_seg_results(U_Net_seg_result_4d_tensor)

        ws_U_Net_seg_result_4d_tensor = self.ws_U_Net_model(img_4d_tensor_torch)
        self.ws_U_Net_seg_result_2d_array = prepare_seg_results(ws_U_Net_seg_result_4d_tensor)

        kmeans_U_Net_seg_result_4d_tensor = self.kmeans_U_Net_model(img_4d_tensor_torch)
        self.kmeans_U_Net_seg_result_2d_array = prepare_seg_results(kmeans_U_Net_seg_result_4d_tensor)

        threshold_U_Net_seg_result_4d_tensor = self.threshold_U_Net_model(img_4d_tensor_torch)
        self.threshold_U_Net_seg_result_2d_array = prepare_seg_results(threshold_U_Net_seg_result_4d_tensor)

    def create_seg_result_mask(self, gt):
        gt[gt == self.tumour_label_value] = self.tumour_pixel_value
        gt_mask = np.ma.masked_where(gt == self.bg_label_value, gt)

        return gt_mask

    def view_segmentation_result(self):
        print('****** Plot Segmentation Results! ******')

        self.process_img_gt_tensor()

        plt.figure(figsize=(12, 8))

        # First row, containing 3 images
        plt.subplot(2, 5, 1)  # original scan
        plt.title('Brain scan')
        plt.axis('off')
        plt.imshow(self.sliced_img_2d_array.T, cmap='gray', origin='lower')

        plt.subplot(2, 5, 2)  # gt
        plt.title('Ground Truth')
        plt.axis('off')
        plt.imshow(self.sliced_gt_2d_array.T, cmap='gray', origin='lower')

        plt.subplot(2, 5, 3)  # box gt
        plt.title('Bounding Box Label')
        plt.axis('off')
        plt.imshow(self.sliced_ws_2d_array.T, cmap='gray', origin='lower')

        plt.subplot(2, 5, 4)  # k Unet
        plt.title('Kmeans Label')
        plt.axis('off')
        plt.imshow(self.sliced_kmeans_2d_array.T, cmap='gray', origin='lower')

        plt.subplot(2, 5, 5)  # Threshold Unet
        plt.title('Threshold Label')
        plt.axis('off')
        plt.imshow(self.sliced_threshold_2d_array.T, cmap='gray', origin='lower')

        plt.subplot(2, 5, 6)  # Brain Image
        plt.title('Brain Image')
        plt.axis('off')
        plt.imshow(self.sliced_img_2d_array.T, cmap='gray', origin='lower')


        plt.subplot(2, 5, 7)  # full Unet
        plt.title('Full U-Net Result')
        plt.axis('off')
        plt.imshow(self.U_Net_seg_result_2d_array.T, cmap='gray', origin='lower')

        plt.subplot(2, 5, 8)  # ws Unet
        plt.title('Weak U-Net Result')
        plt.axis('off')
        plt.imshow(self.ws_U_Net_seg_result_2d_array.T, cmap='gray', origin='lower')

        plt.subplot(2, 5, 9)  # k Unet
        plt.title('Kmeans U-Net Result')
        plt.axis('off')
        plt.imshow(self.kmeans_U_Net_seg_result_2d_array.T, cmap='gray', origin='lower')

        plt.subplot(2, 5, 10)  # k Unet
        plt.title('Threshold U-Net Result')
        plt.axis('off')
        plt.imshow(self.threshold_U_Net_seg_result_2d_array.T, cmap='gray', origin='lower')

        # Adjust spacing between subplots for better visualization
        plt.tight_layout()

        # Show the plot
        plt.show()


visualization = AllModelVisualization()

test_tensor = visualization.test_tensor
train_tensor = visualization.train_tensor
visualization.view_segmentation_result()
