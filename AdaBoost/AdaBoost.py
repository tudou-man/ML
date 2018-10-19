# -*- coding: utf-8 -*-
# author: Caofang
# Time : 2018/10/17 10:54
# @File : AdaBoost.py

import numpy as np
def loadDataSet():
    dataMAT = np.mat([[1. , 2.1],
        [2. , 1.1],
        [1.3, 1. ],
        [1. , 1. ],
        [2. , 1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMAT, classLabels
# dataMAT, classLabels = loadDataSet()
# print (dataMAT, classLabels)

def stumpClassify(dataMAT, dimen, threshVal, threshIneq):
    retArray = np.ones(np.shape(dataMAT)[0], 1)
    # lt：小于，gt；大于；根据阈值进行分类，并将分类结果存储到retArray
    if threshIneq == 'lt':
        retArray[dataMAT[:, dimen] <= threshVal] = -1.0
        # 对每一个数据进行比较，满足条件的的进行赋值，不满足不进行赋值
    else:
        retArray[dataMAT[:, dimen] > threshVal] = -1.0
    return retArray

def buildStump(dataMAT, classLabels):
    m ,n = np.shape(dataMAT)
    num_step = 10.0
    best_stump = {}
    bestClasEst = np.mat(np.zeros(m, 1))
    minError = np.inf
    for j in range(n):

