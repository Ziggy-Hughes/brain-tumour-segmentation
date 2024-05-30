Content:

Folders:

BfcData: holds the Bfc smoothed data.
Code: holds all code files.
Data: holds the unprocessed data.
LogFolder: holds the log outputs.
NoBfcData: holds the non smoothed data.
ProccessedData: holds the data after preprocessing.
saved_model: holds the trained models.

Code:

bbox_coordinate.csv: holds the bounding box data for each slice (x and y coordinates and width and height).
bfc.py: code to smooth data.
config.py: code to get configurations.
config.yml: holds the settings of the programme e.g. learning rate and number of iterations.
evaluate_using_gt.py: code to compare predicted masks to the real gt
evaluation.py: code to calculate performance metrics e.g. accuracy and precision.
hausdorff.py: code to calculate hausdorff distance.
model_visualization.py: code to display images of segmentation results.
move_download_files.py: code to change data file format and location.
move_files.py: code to change data file format and location.
preprocess.py: code to carry out preprocessing e.g. finding kmeans labels.
save_coordinate.py: code to calculate and save bounding boxes.
segmentation_model.py: the deep learning model created using pyTorch.
set_logger.py: code to create log outputs.
test_distances.numbers: holds hausdorff distances with analytics for test data.
test_distances.xlsx: holds hausdorff distances for test data.
train_distances.numbers: holds hausdorff distances with analytics for training data.
train_distances.xlsx: holds hausdorff distances for training data.
train_model.py: code to train and save the deep learning model.
