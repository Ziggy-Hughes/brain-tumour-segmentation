from sklearn.metrics import confusion_matrix
import numpy as np

def calc_performance_metrics(predicted, labels):
    con_mat = confusion_matrix(labels, predicted)

    accuracy_list = []
    error_rate_list = []
    precision_list = []
    sensitivity_list = []
    specificity_list = []
    F1_score_list = []
    dice_list = []
    jaccard_list = []

    N_classes = 2 # background, tumour
    N_total_samples = np.sum(con_mat[:, :])
    for k in range(N_classes):
        TP = float(con_mat[k][k])
        FN = float(np.sum(con_mat[k, : ]) - TP)
        FP = float(np.sum(con_mat[ :, k]) - TP)
        TN = float(N_total_samples - TP - FN - FP)
        accuracy = (TP + TN) / N_total_samples
        error_rate = (FN + FP) / N_total_samples
        precision = TP / (TP + FP + 0.00000000001)
        sensitivity = TP / (TP + FN)
        specificity = TN / (FP + TN + 0.00000000001)
        F1_score = (2 * TP) / (2*TP + FP + FN)
        dice = F1_score
        jaccard = dice / (2-dice)

        accuracy_list.append(accuracy)
        error_rate_list.append(error_rate)
        precision_list.append(precision)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
        F1_score_list.append(F1_score)
        dice_list.append(dice)
        jaccard_list.append(jaccard)

    perf_eva_dict = {}
    perf_eva_dict['Accuracy_List'] = accuracy_list
    perf_eva_dict['Error_Rate_List'] = error_rate_list
    perf_eva_dict['Precision_List'] = precision_list
    perf_eva_dict['Sensitivity_List'] = sensitivity_list
    perf_eva_dict['Specificity_List'] = specificity_list
    perf_eva_dict['F1_Score_List'] = F1_score_list
    perf_eva_dict['Dice_List'] = dice_list
    perf_eva_dict['Jaccard_List'] = jaccard_list

    perf_eva_dict['Average_Accuracy'] = np.mean(accuracy_list)
    perf_eva_dict['Average_Error_Rate'] = np.mean(error_rate_list)
    perf_eva_dict['Average_Precision'] = np.mean(precision_list)
    perf_eva_dict['Average_Sensitivity'] = np.mean(sensitivity_list)
    perf_eva_dict['Average_Specificity'] = np.mean(specificity_list)
    perf_eva_dict['Average_F1_Score'] = np.mean(F1_score_list)
    perf_eva_dict['Average_Dice'] = np.mean(dice_list)
    perf_eva_dict['Average_Jaccard'] = np.mean(jaccard_list)



    return perf_eva_dict
