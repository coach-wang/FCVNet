# -*- coding: utf-8 -*-

import argparse
import os
from utls import prepareData, myGenerator, loadWeight, train512Generator, displayOutput
import numpy as np
import imageio
from model import xNet
import datetime
import time

########## Set seed ##########
from numpy.random import seed
from tensorflow import set_random_seed
#seed(1)
#set_random_seed(2)

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='./Model')
parser.add_argument('--data_root', type=str, default='./hci_dataset')
parser.add_argument('--loadWeight', default=False, help='if load weight')
parser.add_argument('--batchsize', type=int, default=12)
parser.add_argument('--iterations', type=int, default=10000)
config = parser.parse_args()

def run():
    # prepare dataset
    trainDataRoot = config.data_root
    trainDataAll, trainDataLabel = prepareData(trainDataRoot)

    patchSize = 32
    modelLR = 1e-6
    print('LR is %f' % modelLR)

    boolmaskImg4 = imageio.imread('hci_dataset/additional_invalid_area/kitchen/input_Cam040_invalid_ver2.png')
    boolmaskImg6 = imageio.imread('hci_dataset/additional_invalid_area/museum/input_Cam040_invalid_ver2.png')
    boolmaskImg15 = imageio.imread('hci_dataset/additional_invalid_area/vinyl/input_Cam040_invalid_ver2.png')

    boolmaskImg4 = 1.0 * boolmaskImg4[:, :, 3] > 0
    boolmaskImg6 = 1.0 * boolmaskImg6[:, :, 3] > 0
    boolmaskImg15 = 1.0 * boolmaskImg15[:, :, 3] > 0

    # prepare model
    modelTrain = xNet(patchSize, patchSize, modelLR)
    trainGenerator = myGenerator(trainDataAll, trainDataLabel, patchSize, config.batchsize, boolmaskImg4, boolmaskImg6,
                                 boolmaskImg15)

    # load weight
    if config.loadWeight:
        modelTrain, iterStart = loadWeight(modelTrain, config.model_path)
    else:
        iterStart = 0
    f1 = open(txtName, 'a')
    now = datetime.datetime.now()
    f1.write('\n'+str(now)+'\n\n')
    f1.close()

    maxEpoch = 200
    for iterN in range(iterStart, maxEpoch):
        t0 = time.time()
        hist = modelTrain.fit_generator(trainGenerator, steps_per_epoch=config.iterations, epochs=iterN+1, initial_epoch=iterN,
                                        verbose=1, workers=8)
        f1 = open(txtName, 'a')
        f1.write('iter%4d:' % iterN + '\n')
        f1.write(str(hist.history) + '\n')
        f1.close()

        t1 = time.time()

        saveModelPath = '%s_iter%04d' % ('train', iterN)
        modelTrain.save(os.path.join(config.model_path, saveModelPath + '.hdf5'))
        print('model saved!')

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES']="0"

    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    txtName = './log.txt'

    run()
