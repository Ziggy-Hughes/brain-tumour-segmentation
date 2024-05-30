import os
import shutil

paths = []

endPathTrain = "/Users/ziggy/Downloads/ProcessedData/train/"
endPathTest = "/Users/ziggy/Downloads/ProcessedData/test/"
paths.append("/Users/ziggy/Downloads/ProcessedData 2/train/")
paths.append("/Users/ziggy/Downloads/ProcessedData 2/test/")
paths.append("/Users/ziggy/Downloads/ProcessedData 3/train/")
paths.append("/Users/ziggy/Downloads/ProcessedData 3/test/")

trainBool = False
for path in paths:
    trainBool = not trainBool
    for root, dirs, files in os.walk(path):
        if files[0] == ".DS_Store":
            continue
        for i, file in enumerate(files):
            src = os.path.join(root, file)
            if trainBool:
                dst = os.path.join(endPathTrain, file)
            else:
                dst = os.path.join(endPathTest, file)
            shutil.copyfile(src, dst)
            print('*** ' + file + ' Move done ***')

print('*** All moves done ***')
