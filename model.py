import tensorflow as tf
from tensorflow.contrib.keras.api.keras.layers import Input, Activation
from tensorflow.contrib.keras.api.keras.layers import Conv2D, AveragePooling2D, Lambda, GlobalAveragePooling3D, Conv3D
from tensorflow.contrib.keras.api.keras.layers import BatchNormalization
from tensorflow.contrib.keras.api.keras.layers import add, concatenate, multiply, Average
from tensorflow.contrib.keras.api.keras.models import Model
from tensorflow.contrib.keras import backend as K
import numpy as np
from tensorflow.contrib.keras.api.keras.optimizers import Adam
from utls import slicing, newSlice

def mish(x):
    return x * K.tanh(K.softplus(x))

def basicBlock(input, planes, stride, downsample, dilation):
    conv1 = Conv2D(planes, 3, strides=stride, padding='same', dilation_rate=dilation, use_bias=False)(input)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation(mish)(conv1)
    conv2 = Conv2D(planes, 3, padding='same', dilation_rate=dilation, use_bias=False)(conv1)
    conv2 = BatchNormalization()(conv2)
    if downsample is not None:
        input = downsample
    conv2 = add([conv2, input])
    return conv2

def makeLayer(input, planes, blocks, stride, dilation):
    inplanes = 4
    downsample = None
    if stride != 1 or inplanes != planes:
        downsample = Conv2D(planes, 1, strides=stride, padding='same', use_bias=False)(input)
        downsample = BatchNormalization()(downsample)
    layers = basicBlock(input, planes, stride, downsample, dilation)
    for i in range(1, blocks):
        layers = basicBlock(layers, planes, 1, None, dilation)
    return layers

def UpSampling2DBilinear(size):
    return Lambda(lambda x: tf.image.resize_bilinear(x, size, align_corners=True))

def featureExtraction(inputSize1, inputSize2):
    i = Input(shape=(inputSize1, inputSize2, 1))
    firstConv = Conv2D(4, 3, padding='same', use_bias=False)(i)
    firstConv = BatchNormalization()(firstConv)
    firstConv = Activation(mish)(firstConv)
    firstConv = Conv2D(4, 3, padding='same', use_bias=False)(firstConv)
    firstConv = BatchNormalization()(firstConv)
    firstConv = Activation(mish)(firstConv)

    layer1 = makeLayer(firstConv, 4, 2, 1, 1)
    layer2 = makeLayer(layer1, 8, 4, 1, 1)   
    layer3 = makeLayer(layer2, 16, 2, 1, 1)
    layer4 = makeLayer(layer3, 16, 2, 1, 2)

    layer4Size = (layer4.get_shape().as_list()[1], layer4.get_shape().as_list()[2])   # layer4的高和宽(H, W)

    '''FPN module'''
    C2 = Conv2D(64, 3, strides=2, padding='same', use_bias=False)(layer4)
    C2 = BatchNormalization()(C2)
    C2 = Activation(mish)(C2)
    C2 = Conv2D(64, 3, padding='same', use_bias=False)(C2)
    C2 = BatchNormalization()(C2)
    C2 = Activation(mish)(C2)
    C2Size = (C2.get_shape().as_list()[1], C2.get_shape().as_list()[2])

    C3 = Conv2D(128, 3, strides=2, padding='same', use_bias=False)(C2)
    C3 = BatchNormalization()(C3)
    C3 = Activation(mish)(C3)
    C3 = Conv2D(128, 3, padding='same', use_bias=False)(C3)
    C3 = BatchNormalization()(C3)
    C3 = Activation(mish)(C3)
    C3Size = (C3.get_shape().as_list()[1], C3.get_shape().as_list()[2])

    C4 = Conv2D(256, 3, strides=2, padding='same', use_bias=False)(C3)
    C4 = BatchNormalization()(C4)
    C4 = Activation(mish)(C4)
    C4 = Conv2D(256, 3, padding='same', use_bias=False)(C4)
    C4 = BatchNormalization()(C4)
    C4 = Activation(mish)(C4)

    P4 = Conv2D(16, 1, use_bias=False)(C4)    
    P4_1 = UpSampling2DBilinear(C3Size)(P4)
    P3 = Conv2D(16, 1, use_bias=False)(C3)
    P3_2 = Average()([P4_1, P3])
    P3_1 = UpSampling2DBilinear(C2Size)(P3_2)
    P2 = Conv2D(16, 1, use_bias=False)(C2)
    P2_2 = Average()([P3_1, P2])
    P2_1 = UpSampling2DBilinear(layer4Size)(P2_2)
    P1 = Conv2D(16, 1, use_bias=False)(layer4)
    P1_2 = Average()([P2_1, P1])

    P4 = UpSampling2DBilinear(layer4Size)(P4)
    P3 = UpSampling2DBilinear(layer4Size)(P3_2)

    outputFeature = concatenate([P4, P3, P2_1, P1_2])
    lastConv = Conv2D(32, 3, padding='same', use_bias=False)(outputFeature)
    lastConv = BatchNormalization()(lastConv)
    lastConv = Activation(mish)(lastConv)
    lastConv = Conv2D(8, 1, padding='same', use_bias=False)(lastConv)

    model = Model(inputs=[i], outputs=[lastConv])
    return model

def getCostVolume1(inputs):
    shape = K.shape(inputs[0])
    disparityCosts = []
    for d in range(-4, 5):
        tmpList = []
        if d == 0:
            for i in range(len(inputs)):
                tmpList.append(inputs[i])
        else:
            for i in range(len(inputs)):
                if i < 5:  # 0d
                    tensor = tf.contrib.image.translate(inputs[i], [d*(i-4), d*(-2)], 'BILINEAR')
                elif i < 7:  # 90d
                    tensor = tf.contrib.image.translate(inputs[i], [d*(-2), d*(5-i)], 'BILINEAR')
                elif i < 9:
                    tensor = tf.contrib.image.translate(inputs[i], [d*(-2), d*(4-i)], 'BILINEAR')
                elif i < 11:  # 45d
                    tensor = tf.contrib.image.translate(inputs[i], [d*(i-13), d*(9-i)], 'BILINEAR')
                elif i < 13:
                    tensor = tf.contrib.image.translate(inputs[i], [d*(i-12), d*(8-i)], 'BILINEAR')
                elif i < 15:  # m45d
                    tensor = tf.contrib.image.translate(inputs[i], [d*(i-17), d*(i-17)], 'BILINEAR')
                else:
                    tensor = tf.contrib.image.translate(inputs[i], [d*(i-16), d*(i-16)], 'BILINEAR')
                tmpList.append(tensor)
        cost = K.concatenate(tmpList, axis=3)
        disparityCosts.append(cost)  # 9*[B, H, W, 17C]
    costvolume = K.stack(disparityCosts, axis=1)  # [B, 9, H, W, 17C]
    costvolume = K.reshape(costvolume, (shape[0], 9, shape[1], shape[2], 136))
    return costvolume

def getCostVolume2(inputs):
    shape = K.shape(inputs[0])
    disparityCosts = []
    for d in range(-4, 5):
        tmpList = []
        if d == 0:
            for i in range(len(inputs)):
                tmpList.append(inputs[i])
        else:
            for i in range(len(inputs)):
                if i < 5:  # 0d
                    tensor = tf.contrib.image.translate(inputs[i], [d*i, d*(-2)], 'BILINEAR')
                elif i < 7:  # 90d
                    tensor = tf.contrib.image.translate(inputs[i], [d*2, d*(5-i)], 'BILINEAR')
                elif i < 9:
                    tensor = tf.contrib.image.translate(inputs[i], [d*2, d*(4-i)], 'BILINEAR')
                elif i < 11:  # 45d
                    tensor = tf.contrib.image.translate(inputs[i], [d*(i-9), d*(9-i)], 'BILINEAR')
                elif i < 13:
                    tensor = tf.contrib.image.translate(inputs[i], [d*(i-8), d*(8-i)], 'BILINEAR')
                elif i < 15:  # m45d
                    tensor = tf.contrib.image.translate(inputs[i], [d*(i-13), d*(i-17)], 'BILINEAR')
                else:
                    tensor = tf.contrib.image.translate(inputs[i], [d*(i-12), d*(i-16)], 'BILINEAR')
                tmpList.append(tensor)
        cost = K.concatenate(tmpList, axis=3)
        disparityCosts.append(cost)
    costvolume = K.stack(disparityCosts, axis=1)
    costvolume = K.reshape(costvolume, (shape[0], 9, shape[1], shape[2], 136))
    return costvolume

def getCostVolume3(inputs):
    shape = K.shape(inputs[0])
    disparityCosts = []
    for d in range(-4, 5):
        tmpList = []
        if d == 0:
            for i in range(len(inputs)):
                tmpList.append(inputs[i])
        else:
            for i in range(len(inputs)):
                if i < 5:  # 0d
                    tensor = tf.contrib.image.translate(inputs[i], [d*(i-4), d*2], 'BILINEAR')
                elif i < 7:  # 90d
                    tensor = tf.contrib.image.translate(inputs[i], [d*(-2), d*(9-i)], 'BILINEAR')
                elif i < 9:
                    tensor = tf.contrib.image.translate(inputs[i], [d*(-2), d*(8-i)], 'BILINEAR')
                elif i < 11:  # 45d
                    tensor = tf.contrib.image.translate(inputs[i], [d*(i-13), d*(13-i)], 'BILINEAR')
                elif i < 13:
                    tensor = tf.contrib.image.translate(inputs[i], [d*(i-12), d*(12-i)], 'BILINEAR')
                elif i < 15:  # m45d
                    tensor = tf.contrib.image.translate(inputs[i], [d*(i-17), d*(i-13)], 'BILINEAR')
                else:
                    tensor = tf.contrib.image.translate(inputs[i], [d*(i-16), d*(i-12)], 'BILINEAR')
                tmpList.append(tensor)
        cost = K.concatenate(tmpList, axis=3)
        disparityCosts.append(cost)
    costvolume = K.stack(disparityCosts, axis=1)
    costvolume = K.reshape(costvolume, (shape[0], 9, shape[1], shape[2], 136))
    return costvolume

def getCostVolume4(inputs):
    shape = K.shape(inputs[0])
    disparityCosts = []
    for d in range(-4, 5):
        tmpList = []
        if d == 0:
            for i in range(len(inputs)):
                tmpList.append(inputs[i])
        else:
            for i in range(len(inputs)):
                if i < 5:  # 0d
                    tensor = tf.contrib.image.translate(inputs[i], [d*i, d*2], 'BILINEAR')
                elif i < 7:  # 90d
                    tensor = tf.contrib.image.translate(inputs[i], [d*2, d*(9-i)], 'BILINEAR')
                elif i < 9:
                    tensor = tf.contrib.image.translate(inputs[i], [d*2, d*(8-i)], 'BILINEAR')
                elif i < 11:  # 45d
                    tensor = tf.contrib.image.translate(inputs[i], [d*(i-9), d*(13-i)], 'BILINEAR')
                elif i < 13:
                    tensor = tf.contrib.image.translate(inputs[i], [d*(i-8), d*(12-i)], 'BILINEAR')
                elif i < 15:  # m45d
                    tensor = tf.contrib.image.translate(inputs[i], [d*(i-13), d*(i-13)], 'BILINEAR')
                else:
                    tensor = tf.contrib.image.translate(inputs[i], [d*(i-12), d*(i-12)], 'BILINEAR')
                tmpList.append(tensor)
        cost = K.concatenate(tmpList, axis=3)
        disparityCosts.append(cost)
    costvolume = K.stack(disparityCosts, axis=1)
    costvolume = K.reshape(costvolume, (shape[0], 9, shape[1], shape[2], 136))
    return costvolume

def channelAttention(costVolume):
    x = GlobalAveragePooling3D()(costVolume)  
    # x = GlobalAveragePooling3D(keepdims=True)(costVolume)  
    x = Lambda(lambda y: K.expand_dims(K.expand_dims(K.expand_dims(y, 1), 1), 1))(x)  
    x = Conv3D(128, 1, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv3D(17, 1, padding='same')(x)  # [B, 1, 1, 1, 17]
    x = Activation('sigmoid')(x)
    x = Lambda(lambda y: K.repeat_elements(y, 8, -1))(x)  
    return multiply([x, costVolume]) 

def basic(inputSize1, inputSize2):
    i = Input(shape=(9, inputSize1, inputSize2, 136))
    featureN = 64
    dres0 = Conv3D(featureN, 3, padding='same', use_bias=False)(i)
    dres0 = BatchNormalization()(dres0)
    dres0 = Activation(mish)(dres0)
    dres0 = Conv3D(featureN, 3, padding='same', use_bias=False)(dres0)
    dres0 = BatchNormalization()(dres0)
    cost0 = Activation(mish)(dres0)

    dres1 = Conv3D(featureN, 3, padding='same', use_bias=False)(cost0)
    dres1 = BatchNormalization()(dres1)
    dres1 = Activation(mish)(dres1)
    dres1 = Conv3D(featureN, 3, padding='same', use_bias=False)(dres1)
    dres1 = BatchNormalization()(dres1)
    cost0 = add([dres1, cost0])

    dres2 = Conv3D(featureN, 3, padding='same', use_bias=False)(cost0)
    dres2 = BatchNormalization()(dres2)
    dres2 = Activation(mish)(dres2)
    dres2 = Conv3D(featureN, 3, padding='same', use_bias=False)(dres2)
    dres2 = BatchNormalization()(dres2)
    cost0 = add([dres2, cost0])

    classify = Conv3D(featureN, 3, padding='same', use_bias=False)(cost0)
    classify = BatchNormalization()(classify)
    classify1 = Activation(mish)(classify)
    cost = Conv3D(1, 3, padding='same', use_bias=False)(classify1)

    model = Model(inputs=[i], outputs=[cost, classify])
    return model

def refineBasic(costVolume):
    featureN = 64
    dres0 = Conv3D(featureN, 3, padding='same', use_bias=False)(costVolume)
    dres0 = BatchNormalization()(dres0)
    dres0 = Activation(mish)(dres0)
    dres0 = Conv3D(featureN, 3, padding='same', use_bias=False)(dres0)
    dres0 = BatchNormalization()(dres0)
    cost0 = Activation(mish)(dres0)

    dres1 = Conv3D(featureN, 3, padding='same', use_bias=False)(cost0)
    dres1 = BatchNormalization()(dres1)
    dres1 = Activation(mish)(dres1)
    dres1 = Conv3D(featureN, 3, padding='same', use_bias=False)(dres1)
    dres1 = BatchNormalization()(dres1)
    cost0 = add([dres1, cost0])

    dres2 = Conv3D(featureN, 3, padding='same', use_bias=False)(cost0)
    dres2 = BatchNormalization()(dres2)
    dres2 = Activation(mish)(dres2)
    dres2 = Conv3D(featureN, 3, padding='same', use_bias=False)(dres2)
    dres2 = BatchNormalization()(dres2)
    cost0 = add([dres2, cost0])

    classify = Conv3D(featureN, 3, padding='same', use_bias=False)(cost0)
    classify = BatchNormalization()(classify)
    classify = Activation(mish)(classify)
    cost = Conv3D(1, 3, padding='same', use_bias=False)(classify)

    return cost

def disparityRegression(input):
    shape = K.shape(input)
    disparityValues = np.linspace(-4, 4, 9)
    x = K.constant(disparityValues, shape=[9]) 
    x = K.expand_dims(K.expand_dims(K.expand_dims(x, 0), 0), 0) 
    x = tf.tile(x, [shape[0], shape[1], shape[2], 1]) 
    out = K.sum(multiply([input, x]), -1)

    return out

def directionAttention(input):
    x = Conv2D(6, 3, padding='same', use_bias=False)(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(4, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)   # attention [B, H, W, 4]
    return x

def xNet(inputSize1, inputSize2, learningRate):
    '''4 inputs'''
    inputList = []
    for i in range(4):
        inputList.append(Input(shape=(4, inputSize1, inputSize2, 5)))

    '''features'''
    featureExtractionLayer = featureExtraction(inputSize1, inputSize2)  # 需要slice!!
    featureList1 = []
    featureList2 = []
    featureList3 = []
    featureList4 = []
    for i in range(4):
        for j in range(5):
            if i != 0 and j == 2:  # 只取0度的中心子孔径图像, 避免重复
                continue
            featureSlice = Lambda(newSlice, arguments={'index1': i, 'index2': j})(inputList[0])  
            featureSlice = Lambda(lambda x: K.squeeze(x, 1))(featureSlice)
            featureList1.append(featureExtractionLayer(featureSlice))  # 17*[B, H, W, C] 0, 90, 45, 135依次
            featureSlice = Lambda(newSlice, arguments={'index1': i, 'index2': j})(inputList[1])
            featureSlice = Lambda(lambda x: K.squeeze(x, 1))(featureSlice)
            featureList2.append(featureExtractionLayer(featureSlice))
            featureSlice = Lambda(newSlice, arguments={'index1': i, 'index2': j})(inputList[2])
            featureSlice = Lambda(lambda x: K.squeeze(x, 1))(featureSlice)
            featureList3.append(featureExtractionLayer(featureSlice))
            featureSlice = Lambda(newSlice, arguments={'index1': i, 'index2': j})(inputList[3])
            featureSlice = Lambda(lambda x: K.squeeze(x, 1))(featureSlice)
            featureList4.append(featureExtractionLayer(featureSlice))

    '''cost volume building'''
    cv1 = Lambda(getCostVolume1)(featureList1)  # [9, H, W, 17C]
    cv2 = Lambda(getCostVolume2)(featureList2)
    cv3 = Lambda(getCostVolume3)(featureList3)
    cv4 = Lambda(getCostVolume4)(featureList4)

    '''channel attention'''
    cv1 = channelAttention(cv1)  # attention:[B, 1, 1, 1, 17] cv:[B, 9, H, W, 17C]
    cv2 = channelAttention(cv2)
    cv3 = channelAttention(cv3)
    cv4 = channelAttention(cv4)

    '''cost volume aggression'''
    basicLayer = basic(inputSize1, inputSize2)
    cost1, iCost1 = basicLayer(cv1)  # cost:[B, D, H, W, 1], iCost:[B, D, H, W, 64]
    cost2, iCost2 = basicLayer(cv2)
    cost3, iCost3 = basicLayer(cv3)
    cost4, iCost4 = basicLayer(cv4)

    cost1 = Lambda(lambda x: K.permute_dimensions(K.squeeze(x, -1), (0, 2, 3, 1)))(cost1)  # [B, H, W, D]
    cost1_1 = Lambda(lambda x: -x)(cost1)
    pred1 = Activation('softmax')(cost1_1)
    pred1 = Lambda(disparityRegression)(pred1)  # [B, H, W]

    cost2 = Lambda(lambda x: K.permute_dimensions(K.squeeze(x, -1), (0, 2, 3, 1)))(cost2)
    cost2_1 = Lambda(lambda x: -x)(cost2)
    pred2 = Activation('softmax')(cost2_1)
    pred2 = Lambda(disparityRegression)(pred2)

    cost3 = Lambda(lambda x: K.permute_dimensions(K.squeeze(x, -1), (0, 2, 3, 1)))(cost3)
    cost3_1 = Lambda(lambda x: -x)(cost3)
    pred3 = Activation('softmax')(cost3_1)
    pred3 = Lambda(disparityRegression)(pred3)

    cost4 = Lambda(lambda x: K.permute_dimensions(K.squeeze(x, -1), (0, 2, 3, 1)))(cost4)
    cost4_1 = Lambda(lambda x: -x)(cost4)
    pred4 = Activation('softmax')(cost4_1)
    pred4 = Lambda(disparityRegression)(pred4)

    depth1 = Lambda(lambda y: K.expand_dims(K.expand_dims(y, 1), -1))(pred1)  # [B, 1, H, W, 1]
    depth1 = Lambda(lambda y: K.repeat_elements(y, 9, 1))(depth1)  # [B, 9, H, W, 1]

    depth2 = Lambda(lambda y: K.expand_dims(K.expand_dims(y, 1), -1))(pred2)
    depth2 = Lambda(lambda y: K.repeat_elements(y, 9, 1))(depth2)

    depth3 = Lambda(lambda y: K.expand_dims(K.expand_dims(y, 1), -1))(pred3)
    depth3 = Lambda(lambda y: K.repeat_elements(y, 9, 1))(depth3)

    depth4 = Lambda(lambda y: K.expand_dims(K.expand_dims(y, 1), -1))(pred4)
    depth4 = Lambda(lambda y: K.repeat_elements(y, 9, 1))(depth4)

    fuseCost = concatenate([cost1, cost2, cost3, cost4])  # [B, H, W, 4D]
    dirAtt = directionAttention(fuseCost)  # [B, H, W, 4]

    att1 = Lambda(lambda y: K.repeat_elements(K.expand_dims(y[:, :, :, :1], 1), 9, 1))(dirAtt)
    att1 = Lambda(lambda y: K.repeat_elements(y, 64, 4))(att1)  # [B, 9, H, W, 64]

    att2 = Lambda(lambda y: K.repeat_elements(K.expand_dims(y[:, :, :, 1:2], 1), 9, 1))(dirAtt)
    att2 = Lambda(lambda y: K.repeat_elements(y, 64, 4))(att2)

    att3 = Lambda(lambda y: K.repeat_elements(K.expand_dims(y[:, :, :, 2:3], 1), 9, 1))(dirAtt)
    att3 = Lambda(lambda y: K.repeat_elements(y, 64, 4))(att3)

    att4 = Lambda(lambda y: K.repeat_elements(K.expand_dims(y[:, :, :, 3:4], 1), 9, 1))(dirAtt)
    att4 = Lambda(lambda y: K.repeat_elements(y, 64, 4))(att4)

    iCost1 = multiply([iCost1, att1])
    iCost2 = multiply([iCost2, att2])
    iCost3 = multiply([iCost3, att3])
    iCost4 = multiply([iCost4, att4])

    refineCost = Average()([iCost1, iCost2, iCost3, iCost4])
    refineCost = concatenate([depth1, depth2, depth3, depth4, refineCost])  # [B, 9, H, W, 68]

    cost = refineBasic(refineCost)
    cost = Lambda(lambda x: K.permute_dimensions(K.squeeze(x, -1), (0, 2, 3, 1)))(cost)
    cost = Lambda(lambda x: -x)(cost)
    pred = Activation('softmax')(cost)
    pred = Lambda(disparityRegression)(pred)  # 最终深度图

    model = Model(inputs=inputList, outputs=[pred, pred1, pred2, pred3, pred4])  # training
    #model = Model(inputs=inputList, outputs=[pred])  # testing

    model.summary()
    opt = Adam(lr=learningRate)
    model.compile(optimizer=opt, loss='mae', loss_weights=[1.0, 0.25, 0.25, 0.25, 0.25])  # training
    #model.compile(optimizer=opt, loss='mae')  # testing

    return model
