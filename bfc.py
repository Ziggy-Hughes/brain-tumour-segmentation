# conda activate ants_env
# does not split train and test

from nipype.interfaces.ants.segmentation import N4BiasFieldCorrection
import os
import shutil

prevPath = os.path.abspath('..')
dataPath =  os.path.join(prevPath, 'Data')
imgPath = os.path.join(prevPath, 'BfcData/img')
gtPath = os.path.join(prevPath, 'BfcData/gt')

for root, dirs, files in os.walk(dataPath):
    if files[0] == ".DS_Store":
        continue
    # num = files[0].split('_')[2]
    for i, file in enumerate(files):
        if file == ".DS_Store":
            continue
        type = file.split('_')[3]
        if type != "seg.nii.gz":
            n4 = N4BiasFieldCorrection()
            n4.inputs.input_image = os.path.join(root, file)
            n4.inputs.output_image = os.path.join(imgPath, file)
            n4.inputs.dimension = 3
            n4.inputs.n_iterations = [100, 100, 60, 40]
            n4.inputs.shrink_factor = 3
            n4.inputs.convergence_threshold = 1e-4
            n4.inputs.bspline_fitting_distance = 300
            result = n4.run()

            print('*** ' + file + ' N4BiasFieldCorrection done ***')
        else:
            src = os.path.join(root, file)
            dst = os.path.join(gtPath, file)
            shutil.copyfile(src, dst)

            print('*** ' + file + ' Move done ***')
