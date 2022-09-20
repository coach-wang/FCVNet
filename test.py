import os
import numpy as np
from utls import makeInput, write_pfm, read_pfm
from modelTest import xNet
import time
import matplotlib.pyplot as plt
import cv2

if __name__ == '__main__':
    dirOutput = 'output'
    if not os.path.exists(dirOutput):
        os.makedirs(dirOutput)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    LFdir = 'synthetic'

    if (LFdir == 'synthetic'):
        dirLFimages = ['hci_dataset/test/bedroom']

        imageH, imageW = 512, 512

    pathWeight = ['Model/checkPT.hdf5']

    modelLR = 0.001
    modelTest = xNet(imageH, imageW, modelLR)

    for i in pathWeight:
        modelTest.load_weights(i)
        print('load weight %s' % i)

        for imagePath in dirLFimages:
            valList = makeInput(imagePath, imageH, imageW)
            start = time.clock()

            valoutput, attentionTmp = modelTest.predict(valList, batch_size=1)

            runtime = time.clock() - start
            print('runtime: %.5f(s)' % runtime)

            write_pfm(valoutput[0, :, :], dirOutput + '/%s.pfm' % (imagePath.split('/')[-1]))
            print('pfm file saves in %s/%s.pfm' % (dirOutput, imagePath.split('/')[-1]))
        '''
        outputStack = []
        gtStack = []
        for imagePath in dirLFimages:
            output = read_pfm(dirOutput + '/%s.pfm' % (imagePath.split('/')[-1]))
            gt = read_pfm(imagePath + '/gt_disp_lowres.pfm')
            gt490 = gt[15:-15, 15:-15]
            outputStack.append(output)
            gtStack.append(gt490)
        output = np.stack(outputStack, 0)
        gt = np.stack(gtStack, 0)

        output = output[:, 15:-15, 15:-15]
        trainDiff = np.abs(output-gt)
        trainBp = (trainDiff >= 0.07)
        trainMAE100 = 100*np.average(trainDiff)
        trainMSE100 = 100*np.average(np.square(trainDiff))
        trainBadPixelRate = 100*np.average(trainBp)
        print('Model average MAE*100 = %f' % trainMAE100)
        print('Model average MSE*100 = %f' % trainMSE100)
        print('Model average BadPix0.07 = %f' % trainBadPixelRate)
        '''
