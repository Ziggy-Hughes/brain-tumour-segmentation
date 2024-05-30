import os
import shutil
import random


trainImgPath = "/Users/ziggy/Downloads/Project/MyWork/NoBfcData/train/img"
trainGtPath = "/Users/ziggy/Downloads/Project/MyWork/NoBfcData/train/gt"
testImgPath = "/Users/ziggy/Downloads/Project/MyWork/NoBfcData/test/img"
testGtPath = "/Users/ziggy/Downloads/Project/MyWork/NoBfcData/test/gt"

# training 4:1 testing (74 out of 369)
test_indexes = random.sample(range(0, 369), 74)

count = 0
for root, dirs, files in os.walk('/Users/ziggy/Downloads/Project/MyWork/Data/'):
    if files[0] == ".DS_Store":
        continue
    if count in test_indexes:
        dataset = 'test'
    else:
        dataset = 'train'
    count = count + 1
    for i, file in enumerate(files):
        if file == ".DS_Store":
            continue
        if dataset == 'test':
            type = file.split('_')[3]
            if type != "seg.nii.gz":
                src = os.path.join(root, file)
                dst = os.path.join(testImgPath, file)
                shutil.copyfile(src, dst)

                print('*** ' + file + ' Move done ***')
            else:
                src = os.path.join(root, file)
                dst = os.path.join(testGtPath, file)
                shutil.copyfile(src, dst)

                print('*** ' + file + ' Move done ***')
        else:
            type = file.split('_')[3]
            if type != "seg.nii.gz":
                src = os.path.join(root, file)
                dst = os.path.join(trainImgPath, file)
                shutil.copyfile(src, dst)

                print('*** ' + file + ' Move done ***')
            else:
                src = os.path.join(root, file)
                dst = os.path.join(trainGtPath, file)
                shutil.copyfile(src, dst)

                print('*** ' + file + ' Move done ***')
