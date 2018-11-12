# coding=utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

moon_dat = pd.read_excel("moons.xls",header=None)
moon_dat = np.array(moon_dat)

#两点之间的权重为距离的倒数，即距离越远权重越小，生成权重矩阵
def getWeighMat(x):

    xt = np.transpose(x)
    vecProd = np.dot(x,xt)
    sigma = 10**(-3)

    # 使用矩阵计算两点之间的距离
    sqX = x**2

    sumSqX = np.matrix(np.sum(sqX,axis=1))

    sumSqAEx = np.tile(sumSqX.transpose(),(1,vecProd.shape[1]))
    sumSqBEx = np.tile(sumSqX,(vecProd.shape[0],1))

    dist = -(sumSqAEx + sumSqBEx -2*vecProd)
    result = np.exp(dist/(2*sigma))#高斯核函数

    # 对角线元素为自身，为0
    result = result - np.diag(np.diag(result))

    return result

# 计算对角矩阵
def getDiagMat(WMat):

    diag = np.sum(WMat,axis=1)
    diag = np.squeeze(np.array(diag))
    diag = np.diag(diag)

    return diag

# 计算拉普拉斯矩阵
def laplace_array(DMat,WMat):

    return DMat-WMat

#计算得到特征值和特征向量,并选出前K个
def getEigenvectors(Mat,k):

    value,vector = np.linalg.eig(Mat)
    sorted_id = np.argsort(value)
    topk_vecs = vector[:, sorted_id[0:k:1]]

    return topk_vecs

#归一化矩阵
def normalize(Mat):

    topkSqrt = np.array(Mat) * np.array(Mat)

    divmat = np.tile(np.sqrt(np.transpose(sum(np.transpose(topkSqrt)))), (2, 1))
    divmat = np.transpose(divmat)

    Y = np.array(top_k) / divmat

    return Y

#kmeas算法
def kmeans(Mat,K):

    avg = []
    times = 0

    # 初始化均值向量
    for i in range(K):
        avg.append(Mat[i])

    flag = True

    while flag:

        times = times + 1
        m1 = np.zeros(Mat.shape)  # 为第一类
        m2 = np.zeros(Mat.shape)  # 为第二类
        count1 = 0
        count2 = 0

        for i in range(len(Mat)):

            if sum((Mat[i]-avg[0])**2)**0.5 <= sum((Mat[i]-avg[1])**2)**0.5:
                m1[i] = Mat[i]
                count1 = count1 + 1
            else:
                m2[i] = Mat[i]
                count2 = count2 + 1

        sm1 = sum(m1) / count1
        sm2 = sum(m2) / count2

        count = 0

        if (sm1[0] == avg[0][0]) & (sm1[1] == avg[0][1]):
            count = count + 1
        else:
            avg[0] = sm1

        if (sm2[0] == avg[1][0]) & (sm2[1] == avg[1][1]):
            count = count + 1
        else:
            avg[1] = sm2

        if count == 2:
            flag = False

    print("经过"+str(times-1)+"次迭代，完成分类")

    return m1,m2

def getIndex(Mat):

    id = []

    for i in range(len(Mat)):
        if (Mat[i][0] != 0) & (Mat[i][1] != 0):
            id.append(i)
    return id

if __name__ == '__main__':
    W = getWeighMat(moon_dat)
    D = getDiagMat(W)
    L = laplace_array(D, W)
    top_k = getEigenvectors(L, 2)
    Y = normalize(top_k)
    m1, m2 = kmeans(Y, 2)
    result1 = getIndex(m1)
    result1 = np.array(result1)
    result2 = getIndex(m2)
    result2 = np.array(result2)
    a = moon_dat[result1]
    b = moon_dat[result2]

    plt.figure()
    plt.scatter(a[:, 0], a[:, 1], color='b')
    plt.scatter(b[:, 0], b[:, 1], color='r')
    plt.show()