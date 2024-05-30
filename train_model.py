import torch
import segmentation_model
import config
import os
import torchvision
import torch.nn.functional as F
import torch.optim as optim
import evaluation
import numpy as np
import time
import datetime
import set_logger
import logging
from torch.utils.data import Dataset
from  torch.utils.data import DataLoader
from torchvision.transforms import Compose,ToTensor,Normalize

try:
    import _pickle as pickle
except ImportError:
    import pickle

read_configuration = config._config

setlogger = set_logger.Setlogger()
setlogger.create_logger()
logger = logging.getLogger('root')

class CreateDataset(Dataset):
    def __init__(self, is_train=True, transform=None):
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
        ws_label_3d = torch.tensor(self.train_test_data[2], dtype=torch.float32)
        kmeans_3d = torch.tensor(self.train_test_data[3], dtype=torch.float32)
        threshold_3d = torch.tensor(self.train_test_data[4], dtype=torch.float32)

        return img_array_3d, gt_array_3d, ws_label_3d, kmeans_3d, threshold_3d

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

cd = CreateDataset()


class Train(object):

    def __init__(self):
        self.get_param()
        self.build_model()
        self.set_optimizer()

    def build_model(self):
        if self.seg_model_type == 'U-Net':
            self.seg_model = segmentation_model.UNet()
        else:
            self.seg_model = segmentation_model.UNet()

        self.seg_model.to(self.device) # computing devices (cpu or gpu)

    def get_param(self):
        logger.info("****** Get Parameters! *********")
        self.seg_model_type = read_configuration['param_setting']['seg_model_type']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_iteration = read_configuration['param_setting']['training_iteration']
        self.criterion = torch.nn.CrossEntropyLoss()

        self.optimizer_type = read_configuration['param_setting']['optimizer_type'] # can choose 'SGD' or 'Adam'
        self.learning_rate = read_configuration['param_setting']['learning_rate']
        self.SGD_momentum = read_configuration['param_setting']['SGD_momentum']
        self.valid_interval = read_configuration['param_setting']['valid_interval']
        self.batch_size = read_configuration['param_setting']['batch_size']

        self.train_ACC_criterion = read_configuration['save_model_criterion']['train_ACC_criterion']
        self.valid_ACC_criterion = read_configuration['save_model_criterion']['valid_ACC_criterion']
        self.train_F1_criterion = read_configuration['save_model_criterion']['train_F1_criterion']
        self.valid_F1_criterion = read_configuration['save_model_criterion']['valid_F1_criterion']


        logger.info('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        logger.info('if_use_GPU: %s', self.device)
        logger.info('batch_size: %s', self.batch_size)
        logger.info('seg_model_type: %s', self.seg_model_type)
        logger.info('training_iteration: %d', self.training_iteration)
        logger.info('optimizer_type: %s', self.optimizer_type)
        logger.info('loss_function: %s', self.criterion)
        logger.info('learning_rate: %f', self.learning_rate)
        logger.info('valid_interval: %d', self.valid_interval)
        logger.info('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


    def set_optimizer(self):
        logger.info('****** Set Optimizer! ****** ')
        if self.optimizer_type == 'SGD':
            self.optimizer = optim.SGD(self.seg_model.parameters(), lr = self.learning_rate, momentum = self.SGD_momentum)
        elif self.optimizer_type == 'Adam':
            self.optimizer = optim.Adam(self.seg_model.parameters(), lr = self.learning_rate)
        else:
            self.optimizer = optim.Adam(self.seg_model.parameters(), lr = self.learning_rate)




    def perform_validation(self, seg_model, epoch, type, valid_dataset):
        logger.info('===========================VALID ON %s SET============EPOCH = %d ===============================', valid_dataset, epoch)

        if valid_dataset == 'train':
            valid_tensor_batch_list = self.train_dataloader
        else:
            valid_tensor_batch_list = self.test_dataloader

        # Evaluation Metrics
        valid_ACC_list = [] # Accuracy
        valid_ERR_list = [] # Error rate
        valid_SE_list = [] # Sensitivity (Recall)
        valid_SP_list = [] # Specificity
        valid_PC_list = [] # Precision
        valid_F1_list = [] # F1 score
        valid_JAC_list = [] # Jaccard Similarity
        valid_DSC_list = [] # Dice Cofficient

        for j, (valid_img, valid_gt, ws_label, kmeans_label, threshold_label) in enumerate(valid_tensor_batch_list):
            valid_img = valid_img.to(self.device) # load pytorch img tesnor to computing devices (cpu or gpu)
            if type == 'full':
                valid_gt = valid_gt.to(self.device, dtype = torch.int64) # load gt tesnor to computing devices (cpu or gpu)
            elif type == 'bb':
                ws_label = ws_label.to(self.device, dtype = torch.int64) # load gt tesnor to computing devices (cpu or gpu)
            elif type == 'kmeans':
                kmeans_label = kmeans_label.to(self.device, dtype = torch.int64) # load gt tesnor to computing devices (cpu or gpu)
            elif type == 'threshold':
                threshold_label = threshold_label.to(self.device, dtype = torch.int64) # load gt tesnor to computing devices (cpu or gpu)
            else:
                valid_gt = valid_gt.to(self.device, dtype = torch.int64) # load gt tesnor to computing devices (cpu or gpu)
            valid_seg_result = seg_model(valid_img)
            valid_seg_result = valid_seg_result.squeeze(0)
            valid_seg_result_max = torch.max(valid_seg_result, 1)[1]
            valid_seg_result_flat = torch.flatten(valid_seg_result_max)
            if type == 'full':
                valid_label_flat = torch.flatten(valid_gt)
            elif type == 'bb':
                valid_label_flat = torch.flatten(ws_label)
            elif type == 'kmeans':
                valid_label_flat = torch.flatten(kmeans_label)
            elif type == 'threshold':
                valid_label_flat = torch.flatten(threshold_label)
            else:
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

            logger.info('****** epoch = %d, valid on %s set, valid_avg_accuracy = %.8f, valid_avg_error_rate = %.8f, valid_avg_precision = %.8f, valid_avg_sensitivity = %.8f, valid_avg_specificity = %.8f, valid_avg_F1_score = %.8f, valid_avg_dice = %.8f, valid_avg_jaccard = %.8f',
                              epoch, valid_dataset, valid_avg_accuracy, valid_avg_error_rate, valid_avg_precision, valid_avg_sensitivity, valid_avg_specificity, valid_avg_F1_score, valid_avg_Dice, valid_avg_Jaccard)

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

        logger.info('%%%%%% epoch = %d, valid on %s set, valid_ACC = %.8f, valid_ERR = %.8f, valid_PC = %.8f, valid_SE = %.8f, valid_SP = %.8f, valid_F1 = %.8f, valid_DSC = %.8f, valid_JAC = %.8f',
                         epoch, valid_dataset, valid_ACC, valid_ERR, valid_PC, valid_SE, valid_SP, valid_F1, valid_DSC, valid_JAC)

        logger.info('===============================================================================================\n')


    def train(self, type):
        logger.info('****** Train! ******')

        transform_fn = Compose([ToTensor()])
        train_dataset = CreateDataset(is_train=True, transform=transform_fn)
        test_dataset = CreateDataset(is_train=False, transform=transform_fn)
        batch_size = read_configuration['param_setting']['batch_size']

        self.train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=True)

        for epoch in range(self.training_iteration):
            epoch_start_time = time.time()
            self.seg_model.train(True) # set the training mode
            epoch_loss = 0

            for i, (img, gt, ws_label, kmeans_label, threshold_label) in enumerate(self.train_dataloader):
                img = img.to(self.device) # load pytorch img tesnor to computing devices (cpu or gpu)
                if type == 'full':
                    gt = gt.to(self.device, dtype = torch.int64) # load gt tesnor to computing devices (cpu or gpu)
                elif type == 'bb':
                    ws_label = ws_label.to(self.device, dtype = torch.int64) # load bb tesnor to computing devices (cpu or gpu)
                elif type == 'kmeans':
                    kmeans_label = kmeans_label.to(self.device, dtype = torch.int64) # load kmeans label tesnor to computing devices (cpu or gpu)
                elif type == 'threshold':
                    threshold_label = threshold_label.to(self.device, dtype = torch.int64) # load threshold label tesnor to computing devices (cpu or gpu)
                else:
                    gt = gt.to(self.device, dtype = torch.int64) # load gt tesnor to computing devices (cpu or gpu)
                self.optimizer.zero_grad()
                seg_result = self.seg_model(img)
                if type == 'full':
                    label_3d = torch.squeeze(gt, 1)
                elif type == 'bb':
                    label_3d = torch.squeeze(ws_label, 1)
                elif type == 'kmeans':
                    label_3d = torch.squeeze(kmeans_label, 1)
                elif type == 'threshold':
                    label_3d = torch.squeeze(threshold_label, 1)
                else:
                    label_3d = torch.squeeze(gt, 1)
                batch_loss = self.criterion(seg_result, label_3d)
                epoch_loss = epoch_loss + batch_loss.item()
                batch_loss.backward() # backpropagation
                self.optimizer.step() # optimization
                logger.info("****** epoch = %d, batch_id = %d, batch_loss = %.10f", epoch, i, batch_loss.item())
            logger.info("-------EPOCH = %d, TOTAL_EPOCH_LOSS = %.10f -------\n" % (epoch, epoch_loss))

            if (epoch+1) % self.valid_interval == 0:
                print('******', self.valid_interval)
                self.perform_validation(self.seg_model, epoch, type, valid_dataset = 'train')
                self.perform_validation(self.seg_model, epoch, type, valid_dataset = 'test')

        self.save_model(type)

        seg_array = seg_result.detach().cpu().numpy()
        if type == 'full':
            label_array = gt.detach().cpu().numpy()
        elif type == 'bb':
            label_array = ws_label.detach().cpu().numpy()
        elif type == 'kmeans':
            label_array = kmeans_label.detach().cpu().numpy()
        elif type == 'threshold':
            label_array = threshold_label.detach().cpu().numpy()
        else:
            label_array = gt.detach().cpu().numpy()
        seg_flat_array = seg_array.flatten()
        gt_flat_array = label_array.flatten()
        return seg_array, label_array, seg_flat_array, gt_flat_array


    def save_model(self, type):
        prev_folder_path = os.path.abspath('..')
        saved_model_folder_name = 'saved_model'
        saved_model_folder_path = os.path.join(prev_folder_path, saved_model_folder_name)
        if not os.path.exists(saved_model_folder_path):
            os.makedirs(saved_model_folder_path)
        stored_time = time.strftime('%Y%m%d_%H-%M-%S',time.localtime(time.time()))

        saved_model_fname = str(self.seg_model_type) + '_' + type + '_Segmentation_Model_' + str(stored_time) + '.pth'
        saved_model_fpath = os.path.join(saved_model_folder_path, saved_model_fname)

        torch.save(self.seg_model, saved_model_fpath)


#     def check_if_saved_model(self, train_perf_dict, valid_perf_dict):
#         train_ACC = train_perf_dict['ACC']
#         train_F1 = train_perf_dict['F1']
#         valid_ACC = valid_perf_dict['ACC']
#         valid_F1 = valid_perf_dict['F1']
#
#         if (train_ACC > self.train_ACC_criterion) and (valid_ACC > self.valid_ACC_criterion) and \
#             (train_F1 > self.train_F1_criterion) and (valid_F1 > self.valid_F1_criterion):
#             return True
#         else:
#             return False

for type in ['threshold']: # 'full', 'bb', 'kmeans', 'threshold'
    program_start_time = time.time()
    train = Train()
    seg_array, gt_array, seg_flat_array, gt_flat_array = train.train(type)
    program_end_time = time.time()
    program_execution_time = program_end_time - program_start_time
    program_execution_time = datetime.timedelta(seconds = program_execution_time)
    logger.info('The program execution time is: %s', str(program_execution_time))
