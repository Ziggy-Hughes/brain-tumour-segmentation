import torch
import os
import config
import numpy as np
import evaluation
import set_logger
import logging
from torch.utils.data import Dataset
from  torch.utils.data import DataLoader

read_configuration = config._config

try:
    import _pickle as pickle
except ImportError:
    import pickle

setlogger = set_logger.Setlogger()
setlogger.create_logger()
logger = logging.getLogger('root')

class CreateDataset(Dataset):
    def __init__(self, is_train=True):
        self.is_train = is_train

        self.get_dir()
        for file_name in os.listdir(self.train_test_folder_path):
            if file_name != '.DS_Store':
                self.img_name_list.append(file_name)

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, index):
        img_file_name = self.img_name_list[index]

        self.train_test_data = self.load_data(img_file_name)

        img_array_3d = torch.tensor(self.train_test_data[0], dtype=torch.float32)
        gt_array_3d = torch.tensor(self.train_test_data[1], dtype=torch.float32)

        return img_array_3d, gt_array_3d

    def get_dir(self):
        self.img_name_list = []

        prevPath = os.path.abspath('..')
        path = os.path.join(prevPath, 'ProcessedData')

        if self.is_train is True:
            self.train_test_folder_path = os.path.join(path, 'train')
        elif self.is_train is False:
            self.train_test_folder_path = os.path.join(path, 'train')

    def load_data(self, img_file_name):
        img_data_path = os.path.join(self.train_test_folder_path, img_file_name)

        train_test_img_data = np.load(img_data_path, allow_pickle=True)

        return train_test_img_data

class Evaluator(object):
    def __init__(self):
        self.load_train_test_pickel()
        self.load_segmentation_model()

    def load_train_test_pickel(self):
        train_dataset = CreateDataset(is_train=True)
        test_dataset = CreateDataset(is_train=False)
        batch_size = read_configuration['param_setting']['batch_size']

        self.train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=True)

    def load_segmentation_model(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    def perform_validation(self, type, valid_dataset):
        logger.info('===========================VALID ON %s SET==========', valid_dataset)

        if valid_dataset == 'train':
            valid_tensor_batch_list = self.train_dataloader
        else:
            valid_tensor_batch_list = self.test_dataloader

        if type == 'full':
            seg_model = self.U_Net_model
        elif type == 'bb':
            seg_model = self.ws_U_Net_model
        elif type == 'kmeans':
            seg_model = self.kmeans_U_Net_model
        elif type == 'threshold':
            seg_model = self.threshold_U_Net_model
        else:
            seg_model = self.U_Net_model

        # Evaluation Metrics
        valid_ACC_list = [] # Accuracy
        valid_ERR_list = [] # Error rate
        valid_SE_list = [] # Sensitivity (Recall)
        valid_SP_list = [] # Specificity
        valid_PC_list = [] # Precision
        valid_F1_list = [] # F1 score
        valid_JAC_list = [] # Jaccard Similarity
        valid_DSC_list = [] # Dice Cofficient

        for j, (valid_img, valid_gt) in enumerate(valid_tensor_batch_list):
            valid_img = valid_img.to(self.device) # load pytorch img tesnor to computing devices (cpu or gpu)
            valid_gt = valid_gt.to(self.device, dtype = torch.int64)
            valid_seg_result = seg_model(valid_img)
            valid_seg_result = valid_seg_result.squeeze(0)
            valid_seg_result_max = torch.max(valid_seg_result, 1)[1]
            valid_seg_result_flat = torch.flatten(valid_seg_result_max)
            valid_label_flat = torch.flatten(valid_gt)
            valid_seg_result_flat_list = list(valid_seg_result_flat.detach().cpu().numpy())
            valid_label_flat_list = list(valid_label_flat.detach().cpu().numpy())

            valid_perf_eva_dict = evaluation.calc_performance_metrics(valid_seg_result_flat_list, valid_label_flat_list)

            valid_avg_accuracy = valid_perf_eva_dict['Average_Accuracy']
            valid_avg_error_rate = valid_perf_eva_dict['Average_Error_Rate']
            valid_avg_precision = valid_perf_eva_dict['Average_Precision']
            valid_avg_sensitivity = valid_perf_eva_dict['Average_Sensitivity']
            valid_avg_specificity = valid_perf_eva_dict['Average_Specificity']
            valid_avg_F1_score = valid_perf_eva_dict['Average_F1_Score']
            valid_avg_Dice = valid_perf_eva_dict['Average_Dice']
            valid_avg_Jaccard = valid_perf_eva_dict['Average_Jaccard']

            logger.info('****** valid on %s set, valid_avg_accuracy = %.8f, valid_avg_error_rate = %.8f, valid_avg_precision = %.8f, valid_avg_sensitivity = %.8f, valid_avg_specificity = %.8f, valid_avg_F1_score = %.8f, valid_avg_dice = %.8f, valid_avg_jaccard = %.8f',
                              valid_dataset, valid_avg_accuracy, valid_avg_error_rate, valid_avg_precision, valid_avg_sensitivity, valid_avg_specificity, valid_avg_F1_score, valid_avg_Dice, valid_avg_Jaccard)

            valid_ACC_list.append(valid_avg_accuracy)
            valid_ERR_list.append(valid_avg_error_rate)
            valid_PC_list.append(valid_avg_precision)
            valid_SE_list.append(valid_avg_sensitivity)
            valid_SP_list.append(valid_avg_specificity)
            valid_F1_list.append(valid_avg_F1_score)
            valid_DSC_list.append(valid_avg_Dice)
            valid_JAC_list.append(valid_avg_Jaccard)

        valid_ACC = np.mean(valid_ACC_list)
        valid_ERR = np.mean(valid_ERR_list)
        valid_PC = np.mean(valid_PC_list)
        valid_SE = np.mean(valid_SE_list)
        valid_SP = np.mean(valid_SP_list)
        valid_F1 = np.mean(valid_F1_list)
        valid_DSC = np.mean(valid_DSC_list)
        valid_JAC = np.mean(valid_JAC_list)

        valid_perf_dict = {}
        valid_perf_dict['ACC'] = valid_ACC
        valid_perf_dict['ERR'] = valid_ERR
        valid_perf_dict['PC'] = valid_PC
        valid_perf_dict['SE'] = valid_SE
        valid_perf_dict['SP'] = valid_SP
        valid_perf_dict['F1'] = valid_F1
        valid_perf_dict['DSC'] = valid_DSC
        valid_perf_dict['JAC'] = valid_JAC

        logger.info('%%%%%% valid on %s set, valid_ACC = %.8f, valid_ERR = %.8f, valid_PC = %.8f, valid_SE = %.8f, valid_SP = %.8f, valid_F1 = %.8f, valid_DSC = %.8f, valid_JAC = %.8f',
                         valid_dataset, valid_ACC, valid_ERR, valid_PC, valid_SE, valid_SP, valid_F1, valid_DSC, valid_JAC)

        logger.info('===============================================================================================\n')


eval = Evaluator()
for data in ['train', 'test']:
    for type in ['full', 'bb', 'kmeans', 'threshold']:
        eval.perform_validation(type, data)
