# -*- coding: utf-8 -*-
# author: Caofang
# Time : 2018/10/9 21:39
# @File : PCA.py


import numpy as np
import pandas as pd
# import matplotlib
# .use('Qt5Agg')
import matplotlib.pyplot as plt
# 定义一个均值函数
def meanX(dataX):
    return np.mean(dataX, axis=0)

# 编写PCA方法
def pca(XMat, k):
    """
    參数：
        - XMat：传入的是一个numpy的矩阵格式，行表示样本数，列表示特征
        - k：表示取前k个特征值相应的特征向量
    返回值：
        - finalData：參数一指的是返回的低维矩阵，相应于输入參数二
        - reconData：參数二相应的是移动坐标轴后的矩阵
    """
    average = meanX(XMat)
    m, n = np.shape(XMat)
    data_adjust = []
    avgs = np.tile(average,(m,1))
    data_adjust = XMat - avgs
    covX = np.cov(data_adjust.T)    #计算协方差矩阵
    featValue, featVec = np.linalg.eig(covX)   #计算协方差矩阵的特征值和特征向量
    index = np.argsort(-featValue)  #按照大小排序，返回小标
    finalData = []
    if k > n:
        print ('k must lower than feature number')
        return
    else:
        selectVec = np.matrix(featVec.T[index[:k]])
        print (selectVec)
        finalData = data_adjust * selectVec.T
        reconData = (finalData * selectVec) + average
    return finalData, reconData
# 输入文件的每行数据都以\t隔开g
def loadata(datafile):
    return np.array(pd.read_csv(datafile, sep='\t', header=-1)).astype(np.float)

# 可视化结果
def plotBestFit(data1, data2):
    dataArr1 = np.array(data1)
    dataArr2 = np.array(data2)
    m = np.shape(dataArr1)[0]
    x1 = dataArr1[:,0].T
    y1 = dataArr1[:,1].T
    x2 = dataArr2[:,0].T
    y2 = dataArr2[:,1].T

    fig = plt.figure()
    plt.scatter(x1, y1, s=50, c='red', marker='s')
    plt.scatter(x2, y2, s=50, c='blue')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
    # plt.savefig('outfile.png')

if __name__ == "__main__":
    datafile = 'data_PCA.txt'
    XMat = loadata(datafile)
    k = 2
    finalData, reconData = pca(XMat, k)
    plotBestFit(finalData,reconData)