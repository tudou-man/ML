# -*- coding: utf-8 -*-
# author: Caofang
# Time : 2018/10/16 11:18
# @File : Logistic_Regression.py

from numpy import *
import matplotlib.pyplot as plt

# 读取文件，加载数据，特征X， 标签label
def loadDataSet(filename):
    dataMatrix = []
    datalabel = []
    fr = open(filename)
    for line in fr.readlines():
        # print (line)
        # str.strip()去除首位的字符，默认为空格
        # str.split():将字符串按照一定形式划分成一个list
        line_arr = line.strip().split()        # print (float(line_arr[0]))
        dataMatrix.append([float(line_arr[0]),float(line_arr[1])])
        datalabel.append([float(line_arr[2])])
    dataMAT = mat(dataMatrix)
    m = shape(dataMAT)[0]
    dataMAT = hstack((dataMAT, ones((m,1))))
    labelMAT = mat(datalabel)
    return dataMAT, labelMAT

# sigmod函数
def sigmod(x):
    return 1/(1 + exp(-x))

# 梯度上升算法，每次参数迭代时都需要遍历整个数据集
def graAscent(dataMAT, labelMAT):
    m, n = shape(dataMAT)
    w = ones((n,1))
    alpha = 0.001
    num = 500
    for i in range(num):
        error = sigmod(dataMAT*w) - labelMAT
        w = w - alpha * dataMAT.transpose()*error
    return w

#随机梯度上升算法的实现，对于数据量较多的情况下计算量小，但分类效果差
#每次参数迭代时通过一个数据进行运算
def stoGraAscent(dataMAT, labelMAT):
    m, n = shape(dataMAT)
    w = ones((n,1))
    alpha = 0.001
    num = 20
    for i in range(num):
        for j in range(m):
            error = sigmod(dataMAT[j]*w) - labelMAT[j]
            w = w + alpha * dataMAT[j].transpose() * error
    return w

#改进后的随机梯度上升算法
#从两个方面对随机梯度上升算法进行了改进,正确率确实提高了很多
#改进一：对于学习率alpha采用非线性下降的方式使得每次都不一样
#改进二：每次使用一个数据，但是每次随机的选取数据，选过的不在进行选择
def stoGraAscentalpha(dataMAT, labelMAT):
    m, n = shape(dataMAT)
    # print (m,n)
    w = ones((n,1))
    num = 200
    setIndex = set([])
    for i in range(num):
        for j in range(m):
            alpha = 4/(1 + i + j)+0.01
            dataIndex = random.randint(0,m)
            while dataIndex in setIndex:
                setIndex.add(dataIndex)
                dataIndex = random.randint(0, m)
            error = sigmod(dataMAT[dataIndex] * w - labelMAT[dataIndex])
            w = w + dataMAT[dataIndex].transpose() * error
    return w

def plot_func(weight,filename):
    x0_list = []
    y0_list = []
    x1_list = []
    y1_list = []
    f = open(filename,'r')
    for line in f.readlines():
        lineList = line.strip().split()
        if lineList[2] == 1:
            x0_list.append(float(lineList[0]))
            y0_list.append(float(lineList[1]))
        else:
            x1_list.append(float(lineList[0]))
            y1_list.append(float(lineList[1]))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x0_list, y0_list, s=10, c='red')
    ax.scatter(x1_list, y1_list, s=10, c='green')

    xList = []
    yList = []
    x = arange(-3, 3, 0.1)
    for i in range(len(x)):
        xList.append(x[i])
    # y = (-weight[0] - weight[1] * x)/weight[2]
    for j in range(len(x)):
        y = float((-weight[0] - weight[1] * x[j])/weight[2])
        yList.append(y)
    # print (shape(xList), shape(yList))
    # print (shape(weight))
    ax.plot(xList, yList)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

if __name__=='__main__':
    filename = 'train_data.txt'
    dataMAT ,labelMAT = loadDataSet(filename)
    # print (dataMAT)
    # print (labelMAT)
    weight = stoGraAscentalpha(dataMAT, labelMAT)
    # print (weight)
    plot_func(weight, filename)