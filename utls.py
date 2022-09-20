import numpy as np
import os
import imageio
import sys
import tensorflow as tf

dirTrainImages = ['additional/antinous', 'additional/boardgames', 'additional/dishes',   'additional/greek',
                  'additional/kitchen',  'additional/medieval2',  'additional/museum',   'additional/pens',
                  'additional/pillows',  'additional/platonic',   'additional/rosemary', 'additional/table',
                  'additional/tomb',     'additional/tower',      'additional/town',     'additional/vinyl']

'''生成数据'''
def prepareData(dataDir):

    traindataAll = np.zeros((len(dirTrainImages), 512, 512, 9, 9, 3), np.uint8)
    traindataLabel = np.zeros((len(dirTrainImages), 512, 512), np.float32)

    imageID = 0
    for i in dirTrainImages:
        pathInput = os.path.join(dataDir, i)
        for a in range(81):
            try:
                tmp = np.float32(imageio.imread(pathInput+'/input_Cam0%.2d.png' % a))
            except:
                print(pathInput+'/input_Cam0%.2d.png does not exist' % a)
            traindataAll[imageID, :, :, a//9, a-9*(a//9), :] = tmp
            del tmp
        try:
            tmp = np.float32(read_pfm(pathInput+'/gt_disp_lowres.pfm'))
        except:
            print(pathInput+'/gt_disp_lowres.pfm..dose not exist')
        traindataLabel[imageID, :, :] = tmp
        del tmp
        imageID += 1

    return traindataAll, traindataLabel

'''读取pfm图'''
def read_pfm(fpath, expected_identifier="Pf"):
    # PFM format definition: http://netpbm.sourceforge.net/doc/pfm.html

    def _get_next_line(f):
        next_line = f.readline().decode('utf-8').rstrip()
        # ignore comments
        while next_line.startswith('#'):
            next_line = f.readline().rstrip()
        return next_line

    with open(fpath, 'rb') as f:
        #  header
        identifier = _get_next_line(f)
        if identifier != expected_identifier:
            raise Exception('Unknown identifier. Expected: "%s", got: "%s".' % (expected_identifier, identifier))

        try:
            line_dimensions = _get_next_line(f)
            dimensions = line_dimensions.split(' ')
            width = int(dimensions[0].strip())
            height = int(dimensions[1].strip())
        except:
            raise Exception('Could not parse dimensions: "%s". '
                            'Expected "width height", e.g. "512 512".' % line_dimensions)

        try:
            line_scale = _get_next_line(f)
            scale = float(line_scale)
            assert scale != 0
            if scale < 0:
                endianness = "<"
            else:
                endianness = ">"
        except:
            raise Exception('Could not parse max value / endianess information: "%s". '
                            'Should be a non-zero number.' % line_scale)

        try:
            data = np.fromfile(f, "%sf" % endianness)
            data = np.reshape(data, (height, width))
            data = np.flipud(data)
            with np.errstate(invalid="ignore"):
                data *= abs(scale)
        except:
            raise Exception('Invalid binary values. Could not create %dx%d array from input.' % (height, width))

        return data

'''生成验证数据batch'''
def train512Generator(traindata,traindatalabel, patchsize, batchsize, angularViewsetting):
    # patchsize = 512
    traindataBatch = np.zeros((batchsize, patchsize, patchsize, len(angularViewsetting), len(angularViewsetting)), dtype=np.float32)
    traindataLabelBatch = np.zeros((batchsize, patchsize, patchsize))

    for i in range(batchsize):
        R, G, B = 0.299, 0.587, 0.114

        imageID = i

        idxStart, idyStart = 0, 0

        traindataBatch[i, :, :, :, :] = np.squeeze(R*traindata[imageID:imageID+1, idxStart:idxStart+patchsize, idyStart:idyStart+patchsize, :, :, 0].astype('float32')+
                                                   G*traindata[imageID:imageID+1, idxStart:idxStart+patchsize, idyStart:idyStart+patchsize, :, :, 1].astype('float32')+
                                                   B*traindata[imageID:imageID+1, idxStart:idxStart+patchsize, idyStart:idyStart+patchsize, :, :, 2].astype('float32'))

        traindataLabelBatch[i, :, :] = traindatalabel[imageID, idxStart:idxStart+patchsize, idyStart:idyStart+patchsize]

    traindataBatch = np.float32((1/255)*traindataBatch)
    traindataBatch = np.minimum(np.maximum(traindataBatch, 0), 1)

    traindataBatchList = []
    for i in range(traindataBatch.shape[3]):
        for j in range(traindataBatch.shape[4]):
            traindataBatchList.append(np.expand_dims(traindataBatch[:, :, :, i, j], axis=-1))

    return traindataBatchList, traindataLabelBatch

'''生成训练数据batch'''
def trainBatchGenerator(traindata, traindatalabel, patchsize, batchsize, boolmask4, boolmask6, boolmask15):
    traindataBatch1 = np.zeros((batchsize, 4, patchsize, patchsize, 5), dtype=np.float32)
    traindataBatch2 = np.zeros((batchsize, 4, patchsize, patchsize, 5), dtype=np.float32)
    traindataBatch3 = np.zeros((batchsize, 4, patchsize, patchsize, 5), dtype=np.float32)
    traindataBatch4 = np.zeros((batchsize, 4, patchsize, patchsize, 5), dtype=np.float32)
    traindataLabelBatch = np.zeros((batchsize, patchsize, patchsize))

    for i in range(batchsize):
        sumDiff, valid = 0, 0
        while sumDiff < 0.01*patchsize*patchsize or valid < 1:
            rand3color = 0.05 + np.random.rand(3)
            rand3color = rand3color / np.sum(rand3color)
            R, G, B = rand3color[0], rand3color[1], rand3color[2]

            imageSe = np.array([0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14,
                                0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14,
                                0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
            imageId = np.random.choice(imageSe)

            kk = np.random.randint(17)
            if kk < 8:
                scale = 1
            elif kk < 14:
                scale = 2
            else:
                scale = 3

            idxStart = np.random.randint(0, 512-scale*patchsize)
            idyStart = np.random.randint(0, 512-scale*patchsize)
            valid = 1

            if imageId == 4 or imageId == 6 or imageId == 15:
                if imageId == 4:
                    aTmp = boolmask4
                if imageId == 6:
                    aTmp = boolmask6
                if imageId == 15:
                    aTmp = boolmask15
                if np.sum(aTmp[idxStart:idxStart+scale*patchsize:scale, idyStart:idyStart+scale*patchsize:scale]) > 0:
                    valid = 0

            if valid > 0:
                imageCenter = (1/255)*np.squeeze(
                    R*traindata[imageId, idxStart:idxStart+scale*patchsize:scale,
                      idyStart:idyStart+scale*patchsize:scale, 4, 4, 0].astype('float32') +
                    G*traindata[imageId, idxStart:idxStart+scale*patchsize:scale,
                      idyStart:idyStart+scale*patchsize:scale, 4, 4, 1].astype('float32') +
                    B*traindata[imageId, idxStart:idxStart+scale*patchsize:scale,
                      idyStart:idyStart+scale*patchsize:scale, 4, 4, 2].astype('float32'))
                sumDiff = np.sum(np.abs(imageCenter-np.squeeze(imageCenter[int(0.5*patchsize), int(0.5*patchsize)])))

                for k in range(5):
                    traindataBatch1[i, 0, :, :, k] = np.squeeze(
                        R*traindata[imageId:imageId+1, idxStart:idxStart+scale*patchsize:scale,
                          idyStart:idyStart+scale*patchsize:scale, 2, k, 0].astype('float32') +
                        G*traindata[imageId:imageId+1, idxStart:idxStart+scale*patchsize:scale,
                          idyStart:idyStart+scale*patchsize:scale, 2, k, 1].astype('float32') +
                        B*traindata[imageId:imageId+1, idxStart:idxStart+scale*patchsize:scale,
                          idyStart:idyStart+scale*patchsize:scale, 2, k, 2].astype('float32'))  # 0d
                    traindataBatch1[i, 1, :, :, k] = np.squeeze(
                        R*traindata[imageId:imageId+1, idxStart:idxStart+scale*patchsize:scale,
                          idyStart:idyStart+scale*patchsize:scale, 4-k, 2, 0].astype('float32') +
                        G*traindata[imageId:imageId+1, idxStart:idxStart+scale*patchsize:scale,
                          idyStart:idyStart+scale*patchsize:scale, 4-k, 2, 1].astype('float32') +
                        B*traindata[imageId:imageId+1, idxStart:idxStart+scale*patchsize:scale,
                          idyStart:idyStart+scale*patchsize:scale, 4-k, 2, 2].astype('float32'))  # 90d
                    traindataBatch1[i, 2, :, :, k] = np.squeeze(
                        R*traindata[imageId:imageId+1, idxStart:idxStart+scale*patchsize:scale,
                          idyStart:idyStart+scale*patchsize:scale, 4-k, k, 0].astype('float32') +
                        G*traindata[imageId:imageId+1, idxStart:idxStart+scale*patchsize:scale,
                          idyStart:idyStart+scale*patchsize:scale, 4-k, k, 1].astype('float32') +
                        B*traindata[imageId:imageId+1, idxStart:idxStart+scale*patchsize:scale,
                          idyStart:idyStart+scale*patchsize:scale, 4-k, k, 2].astype('float32'))  # 45d
                    traindataBatch1[i, 3, :, :, k] = np.squeeze(
                        R*traindata[imageId:imageId+1, idxStart:idxStart+scale*patchsize:scale,
                          idyStart:idyStart+scale*patchsize:scale, k, k, 0].astype('float32') +
                        G*traindata[imageId:imageId+1, idxStart:idxStart+scale*patchsize:scale,
                          idyStart:idyStart+scale*patchsize:scale, k, k, 1].astype('float32') +
                        B*traindata[imageId:imageId+1, idxStart:idxStart+scale*patchsize:scale,
                          idyStart:idyStart+scale*patchsize:scale, k, k, 2].astype('float32'))  # m45d

                    traindataBatch2[i, 0, :, :, k] = np.squeeze(
                        R*traindata[imageId:imageId+1, idxStart:idxStart+scale*patchsize:scale,
                          idyStart:idyStart+scale*patchsize:scale, 2, k+4, 0].astype('float32') +
                        G*traindata[imageId:imageId+1, idxStart:idxStart+scale*patchsize:scale,
                          idyStart:idyStart+scale*patchsize:scale, 2, k+4, 1].astype('float32') +
                        B*traindata[imageId:imageId+1, idxStart:idxStart+scale*patchsize:scale,
                          idyStart:idyStart+scale*patchsize:scale, 2, k+4, 2].astype('float32'))
                    traindataBatch2[i, 1, :, :, k] = np.squeeze(
                        R*traindata[imageId:imageId+1, idxStart:idxStart+scale*patchsize:scale,
                          idyStart:idyStart+scale*patchsize:scale, 4-k, 6, 0].astype('float32') +
                        G*traindata[imageId:imageId+1, idxStart:idxStart+scale*patchsize:scale,
                          idyStart:idyStart+scale*patchsize:scale, 4-k, 6, 1].astype('float32') +
                        B*traindata[imageId:imageId+1, idxStart:idxStart+scale*patchsize:scale,
                          idyStart:idyStart+scale*patchsize:scale, 4-k, 6, 2].astype('float32'))
                    traindataBatch2[i, 2, :, :, k] = np.squeeze(
                        R*traindata[imageId:imageId+1, idxStart:idxStart+scale*patchsize:scale,
                          idyStart:idyStart+scale*patchsize:scale, 4-k, k+4, 0].astype('float32') +
                        G*traindata[imageId:imageId+1, idxStart:idxStart+scale*patchsize:scale,
                          idyStart:idyStart+scale*patchsize:scale, 4-k, k+4, 1].astype('float32') +
                        B*traindata[imageId:imageId+1, idxStart:idxStart+scale*patchsize:scale,
                          idyStart:idyStart+scale*patchsize:scale, 4-k, k+4, 2].astype('float32'))
                    traindataBatch2[i, 3, :, :, k] = np.squeeze(
                        R*traindata[imageId:imageId+1, idxStart:idxStart+scale*patchsize:scale,
                          idyStart:idyStart+scale*patchsize:scale, k, k+4, 0].astype('float32') +
                        G*traindata[imageId:imageId+1, idxStart:idxStart+scale*patchsize:scale,
                          idyStart:idyStart+scale*patchsize:scale, k, k+4, 1].astype('float32') +
                        B*traindata[imageId:imageId+1, idxStart:idxStart+scale*patchsize:scale,
                          idyStart:idyStart+scale*patchsize:scale, k, k+4, 2].astype('float32'))

                    traindataBatch3[i, 0, :, :, k] = np.squeeze(
                        R*traindata[imageId:imageId+1, idxStart:idxStart+scale*patchsize:scale,
                          idyStart:idyStart+scale*patchsize:scale, 6, k, 0].astype('float32') +
                        G*traindata[imageId:imageId+1, idxStart:idxStart+scale*patchsize:scale,
                          idyStart:idyStart+scale*patchsize:scale, 6, k, 1].astype('float32') +
                        B*traindata[imageId:imageId+1, idxStart:idxStart+scale*patchsize:scale,
                          idyStart:idyStart+scale*patchsize:scale, 6, k, 2].astype('float32'))
                    traindataBatch3[i, 1, :, :, k] = np.squeeze(
                        R*traindata[imageId:imageId+1, idxStart:idxStart+scale*patchsize:scale,
                          idyStart:idyStart+scale*patchsize:scale, 8-k, 2, 0].astype('float32') +
                        G*traindata[imageId:imageId+1, idxStart:idxStart+scale*patchsize:scale,
                          idyStart:idyStart+scale*patchsize:scale, 8-k, 2, 1].astype('float32') +
                        B*traindata[imageId:imageId+1, idxStart:idxStart+scale*patchsize:scale,
                          idyStart:idyStart+scale*patchsize:scale, 8-k, 2, 2].astype('float32'))
                    traindataBatch3[i, 2, :, :, k] = np.squeeze(
                        R*traindata[imageId:imageId+1, idxStart:idxStart+scale*patchsize:scale,
                          idyStart:idyStart+scale*patchsize:scale, 8-k, k, 0].astype('float32') +
                        G*traindata[imageId:imageId+1, idxStart:idxStart+scale*patchsize:scale,
                          idyStart:idyStart+scale*patchsize:scale, 8-k, k, 1].astype('float32') +
                        B*traindata[imageId:imageId+1, idxStart:idxStart+scale*patchsize:scale,
                          idyStart:idyStart+scale*patchsize:scale, 8-k, k, 2].astype('float32'))
                    traindataBatch3[i, 3, :, :, k] = np.squeeze(
                        R*traindata[imageId:imageId+1, idxStart:idxStart+scale*patchsize:scale,
                          idyStart:idyStart+scale*patchsize:scale, 4+k, k, 0].astype('float32') +
                        G*traindata[imageId:imageId+1, idxStart:idxStart+scale*patchsize:scale,
                          idyStart:idyStart+scale*patchsize:scale, 4+k, k, 1].astype('float32') +
                        B*traindata[imageId:imageId+1, idxStart:idxStart+scale*patchsize:scale,
                          idyStart:idyStart+scale*patchsize:scale, 4+k, k, 2].astype('float32'))

                    traindataBatch4[i, 0, :, :, k] = np.squeeze(
                        R*traindata[imageId:imageId+1, idxStart:idxStart+scale*patchsize:scale,
                          idyStart:idyStart+scale*patchsize:scale, 6, k+4, 0].astype('float32') +
                        G*traindata[imageId:imageId+1, idxStart:idxStart+scale*patchsize:scale,
                          idyStart:idyStart+scale*patchsize:scale, 6, k+4, 1].astype('float32') +
                        B*traindata[imageId:imageId+1, idxStart:idxStart+scale*patchsize:scale,
                          idyStart:idyStart+scale*patchsize:scale, 6, k+4, 2].astype('float32'))
                    traindataBatch4[i, 1, :, :, k] = np.squeeze(
                        R*traindata[imageId:imageId+1, idxStart:idxStart+scale*patchsize:scale,
                          idyStart:idyStart+scale*patchsize:scale, 8-k, 6, 0].astype('float32') +
                        G*traindata[imageId:imageId+1, idxStart:idxStart+scale*patchsize:scale,
                          idyStart:idyStart+scale*patchsize:scale, 8-k, 6, 1].astype('float32') +
                        B*traindata[imageId:imageId+1, idxStart:idxStart+scale*patchsize:scale,
                          idyStart:idyStart+scale*patchsize:scale, 8-k, 6, 2].astype('float32'))
                    traindataBatch4[i, 2, :, :, k] = np.squeeze(
                        R*traindata[imageId:imageId+1, idxStart:idxStart+scale*patchsize:scale,
                          idyStart:idyStart+scale*patchsize:scale, 8-k, k+4, 0].astype('float32') +
                        G*traindata[imageId:imageId+1, idxStart:idxStart+scale*patchsize:scale,
                          idyStart:idyStart+scale*patchsize:scale, 8-k, k+4, 1].astype('float32') +
                        B*traindata[imageId:imageId+1, idxStart:idxStart+scale*patchsize:scale,
                          idyStart:idyStart+scale*patchsize:scale, 8-k, k+4, 2].astype('float32'))
                    traindataBatch4[i, 3, :, :, k] = np.squeeze(
                        R*traindata[imageId:imageId+1, idxStart:idxStart+scale*patchsize:scale,
                          idyStart:idyStart+scale*patchsize:scale, k+4, k+4, 0].astype('float32') +
                        G*traindata[imageId:imageId+1, idxStart:idxStart+scale*patchsize:scale,
                          idyStart:idyStart+scale*patchsize:scale, k+4, k+4, 1].astype('float32') +
                        B*traindata[imageId:imageId+1, idxStart:idxStart+scale*patchsize:scale,
                          idyStart:idyStart+scale*patchsize:scale, k+4, k+4, 2].astype('float32'))

                traindataLabelBatch[i, :, :] = (1.0/scale) * traindatalabel[imageId,
                                                             idxStart:idxStart+scale*patchsize:scale,
                                                             idyStart:idyStart+scale*patchsize:scale]

    traindataBatch1 = np.float32((1/255)*traindataBatch1)
    traindataBatch2 = np.float32((1/255)*traindataBatch2)
    traindataBatch3 = np.float32((1/255)*traindataBatch3)
    traindataBatch4 = np.float32((1/255)*traindataBatch4)
    return traindataBatch1, traindataBatch2, traindataBatch3, traindataBatch4, traindataLabelBatch

def dataAugmentation(traindataBatch1, traindataBatch2, traindataBatch3, traindataBatch4, traindataLabelBatch, batchsize):
    for i in range(batchsize):
        grayRand = 0.4*np.random.rand()+0.8
        traindataBatch1[i, :, :, :, :] = pow(traindataBatch1[i, :, :, :, :], grayRand)
        traindataBatch2[i, :, :, :, :] = pow(traindataBatch2[i, :, :, :, :], grayRand)
        traindataBatch3[i, :, :, :, :] = pow(traindataBatch3[i, :, :, :, :], grayRand)
        traindataBatch4[i, :, :, :, :] = pow(traindataBatch4[i, :, :, :, :], grayRand)

        '''transpose'''
        transRand = np.random.randint(0, 2)
        if transRand == 1:
            traindataBatch1T = np.copy(np.transpose(np.squeeze(traindataBatch1[i, :, :, :, :]), (0, 2, 1, 3)))
            traindataBatch2T = np.copy(np.transpose(np.squeeze(traindataBatch3[i, :, :, :, :]), (0, 2, 1, 3)))
            traindataBatch3T = np.copy(np.transpose(np.squeeze(traindataBatch2[i, :, :, :, :]), (0, 2, 1, 3)))
            traindataBatch4T = np.copy(np.transpose(np.squeeze(traindataBatch4[i, :, :, :, :]), (0, 2, 1, 3)))
            traindataLabelBatch[i, :, :] = np.copy(np.transpose(traindataLabelBatch[i, :, :], (1, 0)))

            traindataBatch1[i, 0, :, :, :] = np.copy(traindataBatch1T[1, :, :, ::-1])
            traindataBatch1[i, 1, :, :, :] = np.copy(traindataBatch1T[0, :, :, ::-1])
            traindataBatch1[i, 2, :, :, :] = np.copy(traindataBatch1T[2, :, :, ::-1])
            traindataBatch1[i, 3, :, :, :] = np.copy(traindataBatch1T[3, :, :, :])

            traindataBatch2[i, 0, :, :, :] = np.copy(traindataBatch2T[1, :, :, ::-1])
            traindataBatch2[i, 1, :, :, :] = np.copy(traindataBatch2T[0, :, :, ::-1])
            traindataBatch2[i, 2, :, :, :] = np.copy(traindataBatch2T[2, :, :, ::-1])
            traindataBatch2[i, 3, :, :, :] = np.copy(traindataBatch2T[3, :, :, :])

            traindataBatch3[i, 0, :, :, :] = np.copy(traindataBatch3T[1, :, :, ::-1])
            traindataBatch3[i, 1, :, :, :] = np.copy(traindataBatch3T[0, :, :, ::-1])
            traindataBatch3[i, 2, :, :, :] = np.copy(traindataBatch3T[2, :, :, ::-1])
            traindataBatch3[i, 3, :, :, :] = np.copy(traindataBatch3T[3, :, :, :])

            traindataBatch4[i, 0, :, :, :] = np.copy(traindataBatch4T[1, :, :, ::-1])
            traindataBatch4[i, 1, :, :, :] = np.copy(traindataBatch4T[0, :, :, ::-1])
            traindataBatch4[i, 2, :, :, :] = np.copy(traindataBatch4T[2, :, :, ::-1])
            traindataBatch4[i, 3, :, :, :] = np.copy(traindataBatch4T[3, :, :, :])

        '''rotation'''
        rotRand = np.random.randint(0, 4)
        if rotRand == 1:  # 90d
            traindataBatch1T = np.copy(np.rot90(traindataBatch2[i, :, :, :, :], 1, (1, 2)))
            traindataBatch2T = np.copy(np.rot90(traindataBatch4[i, :, :, :, :], 1, (1, 2)))
            traindataBatch3T = np.copy(np.rot90(traindataBatch1[i, :, :, :, :], 1, (1, 2)))
            traindataBatch4T = np.copy(np.rot90(traindataBatch3[i, :, :, :, :], 1, (1, 2)))
            traindataLabelBatch[i, :, :] = np.copy(np.rot90(traindataLabelBatch[i, :, :], 1, (0, 1)))

            traindataBatch1[i, 0, :, :, :] = np.copy(traindataBatch1T[1, :, :, ::-1])
            traindataBatch1[i, 1, :, :, :] = np.copy(traindataBatch1T[0, :, :, :])
            traindataBatch1[i, 2, :, :, :] = np.copy(traindataBatch1T[3, :, :, :])
            traindataBatch1[i, 3, :, :, :] = np.copy(traindataBatch1T[2, :, :, ::-1])

            traindataBatch2[i, 0, :, :, :] = np.copy(traindataBatch2T[1, :, :, ::-1])
            traindataBatch2[i, 1, :, :, :] = np.copy(traindataBatch2T[0, :, :, :])
            traindataBatch2[i, 2, :, :, :] = np.copy(traindataBatch2T[3, :, :, :])
            traindataBatch2[i, 3, :, :, :] = np.copy(traindataBatch2T[2, :, :, ::-1])

            traindataBatch3[i, 0, :, :, :] = np.copy(traindataBatch3T[1, :, :, ::-1])
            traindataBatch3[i, 1, :, :, :] = np.copy(traindataBatch3T[0, :, :, :])
            traindataBatch3[i, 2, :, :, :] = np.copy(traindataBatch3T[3, :, :, :])
            traindataBatch3[i, 3, :, :, :] = np.copy(traindataBatch3T[2, :, :, ::-1])

            traindataBatch4[i, 0, :, :, :] = np.copy(traindataBatch4T[1, :, :, ::-1])
            traindataBatch4[i, 1, :, :, :] = np.copy(traindataBatch4T[0, :, :, :])
            traindataBatch4[i, 2, :, :, :] = np.copy(traindataBatch4T[3, :, :, :])
            traindataBatch4[i, 3, :, :, :] = np.copy(traindataBatch4T[2, :, :, ::-1])

        if rotRand == 2:  # 180d
            traindataBatch1T = np.copy(np.rot90(traindataBatch4[i, :, :, :, :], 2, (1, 2)))
            traindataBatch2T = np.copy(np.rot90(traindataBatch3[i, :, :, :, :], 2, (1, 2)))
            traindataBatch3T = np.copy(np.rot90(traindataBatch2[i, :, :, :, :], 2, (1, 2)))
            traindataBatch4T = np.copy(np.rot90(traindataBatch1[i, :, :, :, :], 2, (1, 2)))
            traindataLabelBatch[i, :, :] = np.copy(np.rot90(traindataLabelBatch[i, :, :], 2, (0, 1)))

            traindataBatch1[i, 0, :, :, :] = np.copy(traindataBatch1T[0, :, :, ::-1])
            traindataBatch1[i, 1, :, :, :] = np.copy(traindataBatch1T[1, :, :, ::-1])
            traindataBatch1[i, 2, :, :, :] = np.copy(traindataBatch1T[2, :, :, ::-1])
            traindataBatch1[i, 3, :, :, :] = np.copy(traindataBatch1T[3, :, :, ::-1])

            traindataBatch2[i, 0, :, :, :] = np.copy(traindataBatch2T[0, :, :, ::-1])
            traindataBatch2[i, 1, :, :, :] = np.copy(traindataBatch2T[1, :, :, ::-1])
            traindataBatch2[i, 2, :, :, :] = np.copy(traindataBatch2T[2, :, :, ::-1])
            traindataBatch2[i, 3, :, :, :] = np.copy(traindataBatch2T[3, :, :, ::-1])

            traindataBatch3[i, 0, :, :, :] = np.copy(traindataBatch3T[0, :, :, ::-1])
            traindataBatch3[i, 1, :, :, :] = np.copy(traindataBatch3T[1, :, :, ::-1])
            traindataBatch3[i, 2, :, :, :] = np.copy(traindataBatch3T[2, :, :, ::-1])
            traindataBatch3[i, 3, :, :, :] = np.copy(traindataBatch3T[3, :, :, ::-1])

            traindataBatch4[i, 0, :, :, :] = np.copy(traindataBatch4T[0, :, :, ::-1])
            traindataBatch4[i, 1, :, :, :] = np.copy(traindataBatch4T[1, :, :, ::-1])
            traindataBatch4[i, 2, :, :, :] = np.copy(traindataBatch4T[2, :, :, ::-1])
            traindataBatch4[i, 3, :, :, :] = np.copy(traindataBatch4T[3, :, :, ::-1])

        if rotRand == 3:  # 270d
            traindataBatch1T = np.copy(np.rot90(traindataBatch3[i, :, :, :, :], 3, (1, 2)))
            traindataBatch2T = np.copy(np.rot90(traindataBatch1[i, :, :, :, :], 3, (1, 2)))
            traindataBatch3T = np.copy(np.rot90(traindataBatch4[i, :, :, :, :], 3, (1, 2)))
            traindataBatch4T = np.copy(np.rot90(traindataBatch2[i, :, :, :, :], 3, (1, 2)))
            traindataLabelBatch[i, :, :] = np.copy(np.rot90(traindataLabelBatch[i, :, :], 3, (0, 1)))

            traindataBatch1[i, 0, :, :, :] = np.copy(traindataBatch1T[1, :, :, :])
            traindataBatch1[i, 1, :, :, :] = np.copy(traindataBatch1T[0, :, :, ::-1])
            traindataBatch1[i, 2, :, :, :] = np.copy(traindataBatch1T[3, :, :, ::-1])
            traindataBatch1[i, 3, :, :, :] = np.copy(traindataBatch1T[2, :, :, :])

            traindataBatch2[i, 0, :, :, :] = np.copy(traindataBatch2T[1, :, :, :])
            traindataBatch2[i, 1, :, :, :] = np.copy(traindataBatch2T[0, :, :, ::-1])
            traindataBatch2[i, 2, :, :, :] = np.copy(traindataBatch2T[3, :, :, ::-1])
            traindataBatch2[i, 3, :, :, :] = np.copy(traindataBatch2T[2, :, :, :])

            traindataBatch3[i, 0, :, :, :] = np.copy(traindataBatch3T[1, :, :, :])
            traindataBatch3[i, 1, :, :, :] = np.copy(traindataBatch3T[0, :, :, ::-1])
            traindataBatch3[i, 2, :, :, :] = np.copy(traindataBatch3T[3, :, :, ::-1])
            traindataBatch3[i, 3, :, :, :] = np.copy(traindataBatch3T[2, :, :, :])

            traindataBatch4[i, 0, :, :, :] = np.copy(traindataBatch4T[1, :, :, :])
            traindataBatch4[i, 1, :, :, :] = np.copy(traindataBatch4T[0, :, :, ::-1])
            traindataBatch4[i, 2, :, :, :] = np.copy(traindataBatch4T[3, :, :, ::-1])
            traindataBatch4[i, 3, :, :, :] = np.copy(traindataBatch4T[2, :, :, :])

        '''add noise'''
        noiseRand = np.random.randint(0, 12)
        if noiseRand == 0:
            gauss1 = np.random.normal(0.0, np.random.uniform()*np.sqrt(0.2), (traindataBatch1.shape[1],
                                                                              traindataBatch1.shape[2],
                                                                              traindataBatch1.shape[3],
                                                                              traindataBatch1.shape[4]))
            traindataBatch1[i, :, :, :, :] = np.clip(traindataBatch1[i, :, :, :, :] + gauss1, 0.0, 1.0)
            gauss2 = np.random.normal(0.0, np.random.uniform()*np.sqrt(0.2), (traindataBatch2.shape[1],
                                                                              traindataBatch2.shape[2],
                                                                              traindataBatch2.shape[3],
                                                                              traindataBatch2.shape[4]))
            traindataBatch2[i, :, :, :, :] = np.clip(traindataBatch2[i, :, :, :, :] + gauss2, 0.0, 1.0)
            gauss3 = np.random.normal(0.0, np.random.uniform()*np.sqrt(0.2), (traindataBatch3.shape[1],
                                                                              traindataBatch3.shape[2],
                                                                              traindataBatch3.shape[3],
                                                                              traindataBatch3.shape[4]))
            traindataBatch3[i, :, :, :, :] = np.clip(traindataBatch3[i, :, :, :, :] + gauss3, 0.0, 1.0)
            gauss4 = np.random.normal(0.0, np.random.uniform()*np.sqrt(0.2), (traindataBatch4.shape[1],
                                                                              traindataBatch4.shape[2],
                                                                              traindataBatch4.shape[3],
                                                                              traindataBatch4.shape[4]))
            traindataBatch4[i, :, :, :, :] = np.clip(traindataBatch4[i, :, :, :, :] + gauss4, 0.0, 1.0)

    return traindataBatch1, traindataBatch2, traindataBatch3, traindataBatch4, traindataLabelBatch

def myGenerator(traindata, traindataLabel, patchsize, batchsize, boolmask4, boolmask6, boolmask15):
    while 1:
        (traindataBatch1, traindataBatch2, traindataBatch3, traindataBatch4, traindataLabelBatch) = trainBatchGenerator(traindata, traindataLabel, patchsize, batchsize, boolmask4, boolmask6, boolmask15)
        (traindataBatch1, traindataBatch2, traindataBatch3, traindataBatch4, traindataLabelBatch) = dataAugmentation(traindataBatch1, traindataBatch2, traindataBatch3, traindataBatch4, traindataLabelBatch, batchsize)
        traindataBatchList = [traindataBatch1, traindataBatch2, traindataBatch3, traindataBatch4]
        yield (traindataBatchList, [traindataLabelBatch, traindataLabelBatch, traindataLabelBatch, traindataLabelBatch, traindataLabelBatch])

def loadWeight(model, modelPath):
    listName = os.listdir(modelPath)
    listName.sort()
    ckpName = listName[-1]
    idx = ckpName.find("iter")
    iterN = int(ckpName[idx+4:idx+8])+1
    model.load_weights(os.path.join(modelPath, ckpName))
    print("Weights loaded from the checkpoint %s" % ckpName)
    return model, iterN

def displayOutput(trainOutput, trainLabel, iter00, directorySave):
    sz = len(trainLabel)
    trainOutput = np.squeeze(trainOutput)
    trainDiff = np.abs(trainOutput-trainLabel)
    trainBp = (trainDiff >= 0.07)
    trainOutputAll = np.zeros((2*512, sz*512), np.uint8)
    trainOutputAll[0:512, :] = np.uint8(25*np.reshape(np.transpose(trainLabel, (1, 0, 2)), (512, sz*512))+100)
    trainOutputAll[512:2*512, :] = np.uint8(25*np.reshape(np.transpose(trainOutput, (1, 0, 2)), (512, sz*512))+100)
    imageio.imsave(directorySave+'/val_iter%05d.ipg' % (iter00), np.squeeze(trainOutputAll))
    return trainDiff, trainBp

def makeEPIInput(imagePath, seq1, imageH, imageW):
    traindataTmp = np.zeros((1, 4, imageH, imageW, 5), dtype=np.float32)
    RGB = [0.299, 0.587, 0.114]
    if len(imagePath) == 1:
        imagePath = imagePath[0]

    i = 0
    for seq in seq1[0]:
        tmp = np.float32(imageio.imread(imagePath+'/input_Cam0%.2d.png' % seq))
        traindataTmp[0, 0, :, :, i] = (RGB[0]*tmp[:, :, 0]+RGB[1]*tmp[:, :, 1]+RGB[2]*tmp[:, :, 2])/255
        i += 1

    i = 0
    for seq in seq1[1]:
        tmp = np.float32(imageio.imread(imagePath+'/input_Cam0%.2d.png' % seq))
        traindataTmp[0, 1, :, :, i] = (RGB[0]*tmp[:, :, 0]+RGB[1]*tmp[:, :, 1]+RGB[2]*tmp[:, :, 2])/255
        i += 1

    i = 0
    for seq in seq1[2]:
        tmp = np.float32(imageio.imread(imagePath+'/input_Cam0%.2d.png' % seq))
        traindataTmp[0, 2, :, :, i] = (RGB[0]*tmp[:, :, 0]+RGB[1]*tmp[:, :, 1]+RGB[2]*tmp[:, :, 2])/255
        i += 1

    i = 0
    for seq in seq1[3]:
        tmp = np.float32(imageio.imread(imagePath+'/input_Cam0%.2d.png' % seq))
        traindataTmp[0, 3, :, :, i] = (RGB[0]*tmp[:, :, 0]+RGB[1]*tmp[:, :, 1]+RGB[2]*tmp[:, :, 2])/255
        i += 1

    return traindataTmp

def makeInput(imagePath, imageH, imageW):
    seq1_1 = list(range(18, 23, 1)[0:5:])  # 0d:[18, 19, 20, 21, 22]
    seq1_2 = list(range(2, 47, 9)[::-1][0:5:])  # 90d:[38, 29, 20, 11, 2]
    seq1_3 = list(range(4, 44, 8)[::-1][0:5:])  # 45d:[36, 28, 20, 12, 4]
    seq1_4 = list(range(0, 50, 10)[0:5:])  # m45d:[0, 10, 20, 30, 40]
    seq1 = [seq1_1, seq1_2, seq1_3, seq1_4]

    seq2_1 = list(range(22, 27, 1)[0:5:])  # 0d:[22, 23, 24, 25, 26]
    seq2_2 = list(range(6, 51, 9)[::-1][0:5:])  # 90d:[42, 33, 24, 15, 6]
    seq2_3 = list(range(8, 48, 8)[::-1][0:5:])  # 45d:[40, 32, 24, 16, 8]
    seq2_4 = list(range(4, 54, 10)[0:5:])  # m45d:[4, 14, 24, 34, 44]
    seq2 = [seq2_1, seq2_2, seq2_3, seq2_4]

    seq3_1 = list(range(54, 59, 1)[0:5:])  # 0d:[54, 55, 56, 57, 58]
    seq3_2 = list(range(38, 83, 9)[::-1][0:5:])  # 90d:[74, 65, 56, 47, 38]
    seq3_3 = list(range(40, 80, 8)[::-1][0:5:])  # 45d:[72, 64, 56, 48, 40]
    seq3_4 = list(range(36, 86, 10)[0:5:])  # m45d:[36, 46, 56, 66, 76]
    seq3 = [seq3_1, seq3_2, seq3_3, seq3_4]

    seq4_1 = list(range(58, 63, 1)[0:5:])  # 0d:[58, 59, 60, 61, 62]
    seq4_2 = list(range(42, 87, 9)[::-1][0:5:])  # 90d:[78, 69, 60, 51, 42]
    seq4_3 = list(range(44, 84, 8)[::-1][0:5:])  # 45d:[76, 68, 60, 52, 44]
    seq4_4 = list(range(40, 90, 10)[0:5:])  # m45d:[40, 50, 60, 70, 80]
    seq4 = [seq4_1, seq4_2, seq4_3, seq4_4]

    outputList = []

    outputList.append(makeEPIInput(imagePath, seq1, imageH, imageW))
    outputList.append(makeEPIInput(imagePath, seq2, imageH, imageW))
    outputList.append(makeEPIInput(imagePath, seq3, imageH, imageW))
    outputList.append(makeEPIInput(imagePath, seq4, imageH, imageW))

    return outputList

def write_pfm(data, fpath, scale=1, file_identifier=b'Pf', dtype="float32"):
    # PFM format definition: http://netpbm.sourceforge.net/doc/pfm.html

    data = np.flipud(data)
    height, width = np.shape(data)[:2]
    values = np.ndarray.flatten(np.asarray(data, dtype=dtype))
    endianess = data.dtype.byteorder
    print(endianess)

    if endianess == '<' or (endianess == '=' and sys.byteorder == 'little'):
        scale *= -1

    with open(fpath, 'wb') as file:
        file.write((file_identifier))
        file.write(('\n%d %d\n' % (width, height)).encode())
        file.write(('%d\n' % scale).encode())

        file.write(values)

def slicing(x, index, indexEnd=None, interval=1, split=1):
    if indexEnd is None:
        return x[..., index:index+1:interval]
    else:
        if split > 1:
            xN = tf.split(x, split, axis=-1)
            return tf.concat(xN[index:indexEnd:interval], axis=4)
        else:
            return x[..., index:indexEnd:interval]

def newSlice(x, index1, index2):
    return x[:, index1:index1+1:1, :, :, index2:index2+1:1]
    #return x[index1:index1+1:1, :, :, index2:index2+1:1]
