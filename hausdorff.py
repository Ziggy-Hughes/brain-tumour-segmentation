import torch
import os
import config
from scipy.spatial.distance import directed_hausdorff
import pandas as pd
import numpy as np
read_configuration = config._config

try:
    import _pickle as pickle
except ImportError:
    import pickle

# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def process_img_gt_tensor(tensor_batch_list):
    prev_folder_path = os.path.abspath('..')

    full_model_name = read_configuration['full_model_name']
    U_Net_model_path = prev_folder_path +'/saved_model/'+ full_model_name

    ws_model_name = read_configuration['ws_model_name']
    ws_U_Net_model_path = prev_folder_path +'/saved_model/'+ ws_model_name

    kmeans_model_name = read_configuration['kmeans_model_name']
    kmeans_U_Net_model_path = prev_folder_path +'/saved_model/'+ kmeans_model_name

    threshold_model_name = read_configuration['threshold_model_name']
    threshold_U_Net_model_path = prev_folder_path +'/saved_model/'+ threshold_model_name

    U_Net_model = torch.load(U_Net_model_path, map_location='cpu')
    U_Net_model.eval()

    ws_U_Net_model = torch.load(ws_U_Net_model_path, map_location='cpu')
    ws_U_Net_model.eval()

    kmeans_U_Net_model = torch.load(kmeans_U_Net_model_path, map_location='cpu')
    kmeans_U_Net_model.eval()

    threshold_U_Net_model = torch.load(threshold_U_Net_model_path, map_location='cpu')
    threshold_U_Net_model.eval()

    def prepare_seg_results(seg_result_4d_tensor):
        seg_result_3d_tensor = seg_result_4d_tensor.squeeze(0)

        seg_result_max_2d_tensor = torch.max(seg_result_3d_tensor, 0)[1]
        seg_result_2d_array = seg_result_max_2d_tensor.detach().cpu().numpy()

        return seg_result_2d_array

    img_3d_tensor = tensor_batch_list[0]
    gt_3d_tensor =  tensor_batch_list[1]
    ws_3d_tensor =  tensor_batch_list[2]
    kmeans_3d_tensor =  tensor_batch_list[3]
    threshold_3d_tensor =  tensor_batch_list[4]

    sliced_img_2d_array = img_3d_tensor[0]
    sliced_gt_2d_array = gt_3d_tensor[0]
    sliced_ws_2d_array = ws_3d_tensor[0]
    sliced_kmeans_2d_array = kmeans_3d_tensor[0]
    sliced_threshold_2d_array = threshold_3d_tensor[0]

    img_4d_tensor = np.expand_dims(img_3d_tensor, axis=0)

    img_4d_tensor_torch = torch.tensor(img_4d_tensor, dtype=torch.float32)

    U_Net_seg_result_4d_tensor =  U_Net_model(img_4d_tensor_torch)
    U_Net_seg_result_2d_array = prepare_seg_results(U_Net_seg_result_4d_tensor)

    ws_U_Net_seg_result_4d_tensor =  ws_U_Net_model(img_4d_tensor_torch)
    ws_U_Net_seg_result_2d_array = prepare_seg_results(ws_U_Net_seg_result_4d_tensor)

    kmeans_U_Net_seg_result_4d_tensor =  kmeans_U_Net_model(img_4d_tensor_torch)
    kmeans_U_Net_seg_result_2d_array = prepare_seg_results(kmeans_U_Net_seg_result_4d_tensor)

    threshold_U_Net_seg_result_4d_tensor =  threshold_U_Net_model(img_4d_tensor_torch)
    threshold_U_Net_seg_result_2d_array = prepare_seg_results(threshold_U_Net_seg_result_4d_tensor)

    return sliced_img_2d_array, sliced_gt_2d_array, sliced_ws_2d_array, sliced_kmeans_2d_array, sliced_threshold_2d_array, U_Net_seg_result_2d_array, ws_U_Net_seg_result_2d_array,  kmeans_U_Net_seg_result_2d_array, threshold_U_Net_seg_result_2d_array

def hausdorff_distance(ground_truth, model_result):
    # Convert the arrays to coordinate points (x, y) with non-zero values
    gt_points = np.argwhere(ground_truth != 0)
    model_points = np.argwhere(model_result != 0)

    # Calculate the directed Hausdorff distance
    hausdorff_distance_gt_to_model = directed_hausdorff(gt_points, model_points)[0]
    hausdorff_distance_model_to_gt = directed_hausdorff(model_points, gt_points)[0]

    # Take the maximum of the two directed distances
    hausdorff_distance = max(hausdorff_distance_gt_to_model, hausdorff_distance_model_to_gt)

    return hausdorff_distance

def calculate(dataset):
    prevPath = os.path.abspath('..')
    folder_path = os.path.join(prevPath, 'ProcessedData')
    folder_path = os.path.join(folder_path, dataset)

    file_name_list = os.listdir(folder_path)
    file_name_list.sort()

    # Lists to store the calculated distances
    file_names = []

    full_seg_distances = []
    ws_seg_distances = []
    km_seg_distances = []
    thr_seg_distances = []

    ws_label_distances = []
    km_label_distances = []
    thr_label_distances = []

    count = 0
    for file_name in file_name_list:
        if file_name.endswith('.npy'):  # Make sure to filter only the desired data files
            file_path = os.path.join(folder_path, file_name)

            data = np.load(file_path, allow_pickle=True)
            sliced_img_2d_array, sliced_gt_2d_array, sliced_ws_2d_array, sliced_kmeans_2d_array, sliced_threshold_2d_array, U_Net_seg_result_2d_array, ws_U_Net_seg_result_2d_array, kmeans_U_Net_seg_result_2d_array, threshold_U_Net_seg_result_2d_array = process_img_gt_tensor(data)

            # Calculate the distances
            full_seg_distance = hausdorff_distance(sliced_gt_2d_array, U_Net_seg_result_2d_array)
            ws_seg_distance = hausdorff_distance(sliced_gt_2d_array, ws_U_Net_seg_result_2d_array)
            km_seg_distance = hausdorff_distance(sliced_gt_2d_array, kmeans_U_Net_seg_result_2d_array)
            thr_seg_distance = hausdorff_distance(sliced_gt_2d_array, threshold_U_Net_seg_result_2d_array)


            ws_label_distance = hausdorff_distance(sliced_gt_2d_array, sliced_ws_2d_array)
            km_label_distance = hausdorff_distance(sliced_gt_2d_array, sliced_kmeans_2d_array)
            thr_label_distance = hausdorff_distance(sliced_gt_2d_array, sliced_threshold_2d_array)

            # Append the distances to the respective lists
            file_names.append(file_name)
            full_seg_distances.append(full_seg_distance)
            ws_seg_distances.append(ws_seg_distance)
            km_seg_distances.append(km_seg_distance)
            thr_seg_distances.append(thr_seg_distance)

            ws_label_distances.append(ws_label_distance)
            km_label_distances.append(km_label_distance)
            thr_label_distances.append(thr_label_distance)
            count = count+1
            print('***', count,'***',file_name)

    # Create a dictionary with the lists
    data_dict = {
        'file_name': file_names,
        'full_seg_distance': full_seg_distances,
        'ws_seg_distance': ws_seg_distances,
        'km_seg_distance': km_seg_distances,
        'thr_seg_distance': thr_seg_distances,
        'ws_label_distance': ws_label_distances,
        'km_label_distance': km_label_distances,
        'thr_label_distance': thr_label_distances
    }

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data_dict)

    # Specify the file path for the Excel file
    excel_file_path = dataset + '_distances.xlsx'

    # Save the DataFrame to the Excel file
    df.to_excel(excel_file_path, index=False)

for dataset in ['test', 'train']:
    calculate(dataset)
