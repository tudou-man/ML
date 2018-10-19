# -*- coding: utf-8 -*-
# author: Caofang
# Time : 2018/10/10 20:23
# @File : SVM.py

import numpy as np
import pandas as pd
import random
# 读取数据
def loadDataSet(filename):
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        # print (line)
        # str.strip()去除首位的字符，默认为空格
        # str.split():将字符串按照一定形式划分成一个list
        line_arr = line.strip().split()        # print (float(line_arr[0]))
        dataMat.append([float(line_arr[0]),float(line_arr[1])])
        labelMat.append([float(line_arr[2])])
    return dataMat, labelMat
dataMat, labelMat = loadDataSet('train_data.txt')
print (type(dataMat))
# 0~m之间产生一个不是i的整数
def selectrand(i,m):
    j = i
    while (j==i):
        j = int(random.uniform(0,m))
    return j

# 保证x在L和H之间
def clipx(x,H,L):
    if x > H:
        x = H
    else:
        x = L
    return x

# 定义核函数，输入参数
def kernelTrans(X, A, kTup):
    m, n = np.shape(X)
    K = np.mat(np.zeros(m,1))
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            dataRow = X[j, :] - A
            K[j] = dataRow * dataRow.T
        K = np.exp(-kTup[1]*K)
    else:
        raise NameError(' That Kernel is not recognized')
    return K

# 定义类，方便储存数据
class opStruct:
    def __init__(self, datain, datalabels, C, toler, kTup):
        self.X = datain     #数据特征
        self.labelMat = datalabels    #数据类型
        self.C = C      #软间隔
        self.tol = toler    #阈值
        self.m = np.shape(datain)[0]
        self.alphas = np.zeros((self.m, 1))
        self.b = 0
        self.eCache = np.mat(np.zeros(self.m, 2))
        self.K = np.mat(np.zeros(self.m, self.m))
        for i in range(self.m):
            K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)

#
def calcEi(oS, i):
    g = float(np.multiply(oS.alphas,oS.labelMat).T * oS.K[:,i] + oS.b)
    Ei = g - float(oS.labelMat[i])
    return Ei

# 随机选取alpha_j，并返回其E值
def selectJ(i,oS,Ei):
    '''
    在alpha1确定之后优化alpha2
    内层循环启发方式
    :param i: 下标为i的数据索引
    :param oS: 数据结构
    :param Ei: 下标为i的数据误差
    :return: 
    '''

    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1,Ei]
    valideCachelist = np.nonzero(oS.eCache[:,0].A)[0]
    if len(valideCachelist) > 1:
        for k in valideCachelist:
            if k == i: continue
            Ek = calcEi(oS,k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK,Ej
    else:
        j = selectJ(i, oS.m)
        Ej = calcEi(oS, j)
    return j, Ej

# 更新os数据
def updataEi(oS, i):
    Ei = calcEi(oS, i)
    oS.eCache[k] = [1,Ei]

def innerL(i, oS):
    Ei = calcEi(oS, i)
    oS.eChe[k] = [1,Ei]
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alpha[i] < oS.C)) or \
            ((oS.labelMat[i] * Ei > oS.tol) and (oS.alpha[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)
        # 保存更新之前的alphai和alphaj
        alpha_i_old = oS.alpha[i].copy()
        alpha_j_old = oS.alpha[j].copy()
        if labelMat[i] == labelMat[j]:
            L = max(0, alpha_j_old - alpha_i_old)
            H = min(oS.C, oS.C + alpha_j_old - alpha_i_old)
        else:
            L = max(0, alpha_i_old + alpha_j_old - oS.C)
            H = min(0, alpha_j_old + alpha_i_old)
        if L == H:
            print ('L==H')
            return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i, i] - oS.K[j,j]
        if eta >= 0:
            print ('eta >= 0')
            return 0
        oS.alpha[j] += oS.labelMat[j]*(Ei - Ej)/eta
        oS.alpha[j] = clipx(oS.alpha[j],H,L)
        updataEi(oS, j)
        if (abs(oS.alpha[j] - alpha_j_old) < oS.tol):
            print ('j not moving enoough')
            return 0
        b1 = oS.b - Ei - oS.labelMat[i]*oS.K[i,i]*(oS.alpha[i]-alpha_i_old)\
            - oS.labelMat[j]*oS.K[j,i]*(oS.alpha[j] - alpha_j_old) + oS.b
        b2 = oS.b - Ej - oS.labelMat[i]*oS.K[i,j]*(oS.alpha[i]-alpha_i_old)\
            - oS.labelMat[j]*oS.K[j,j]*(oS.alpha[j] - alpha_j_old)
        if (0 < oS.alpha[i] < oS.C):
            oS.b = b1
        elif (0 < oS.alpha[j] < oS.C):
            oS.b = b2
        else:
            oS.b = (b1 + b2)/2
        return 1
    else:
        return 0

# SMO函数，用于加速求解出alpha
def smoP(datain, datalabels, C, toler, maxiter, kTup = ('lin', 0)):
    oS = opStruct(np.mat(datain), np.mat(datalabels).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxiter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i,oS)
                print ('fullSet, iter: %d i:%d, pairs changed %d'% (iter, i, alphaPairsChanged))
            iter += 1
        else:
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:  # 遍历非边界的数据
                alphaPairsChanged += innerL(i, oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d" % iter)
        return oS.b, oS.alphas

def testRbf(data_train, data_test):
    dataArr, labelArr = loadDataSet(data_train)  # 读取训练数据
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', 1.3))  # 通过SMO算法得到b和alpha
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas)[0]  # 选取不为0数据的行数（也就是支持向量）
    sVs = datMat[svInd]  # 支持向量的特征数据
    labelSV = labelMat[svInd]  # 支持向量的类别（1或-1）
    print("there are %d Support Vectors" % np.shape(sVs)[0])  # 打印出共有多少的支持向量
    m, n = np.shape(datMat)  # 训练数据的行列数
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', 1.3))  # 将支持向量转化为核函数
        predict = kernelEval.T * np.multiply(labelSV, alphas[
            svInd]) + b  # 这一行的预测结果（代码来源于《统计学习方法》p133里面最后用于预测的公式）注意最后确定的分离平面只有那些支持向量决定。
        if np.sign(predict) != np.sign(labelArr[i]):  # sign函数 -1 if x < 0, 0 if x==0, 1 if x > 0
            errorCount += 1
    print("the training error rate is: %f" % (float(errorCount) / m))  # 打印出错误率
    dataArr_test, labelArr_test = loadDataSet(data_test)  # 读取测试数据
    errorCount_test = 0
    datMat_test = np.mat(dataArr_test)
    labelMat = np.mat(labelArr_test).transpose()
    m, n = np.shape(datMat_test)
    for i in range(m):  # 在测试数据上检验错误率
        kernelEval = kernelTrans(sVs, datMat_test[i, :], ('rbf', 1.3))
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr_test[i]):
            errorCount_test += 1
    print("the test error rate is: %f" % (float(errorCount_test) / m))

def main():
    filename_train = 'train_data.txt'
    filename_test = 'test_data.txt'
    testRbf(filename_train, filename_test)

if __name__=='__main__':
    main()