# coding=utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

moon_dat = pd.read_excel("moons.xls",header=None)
moon_dat = np.array(moon_dat)
moon_x = moon_dat[:,0]
moon_y = moon_dat[:,1]

#高斯核函数（2维）
def gaussian_2d(x1, y1, x2, y2, ga):
    gau = np.exp(-ga*((x1-x2)**2 + (y1-y2)**2))

    return gau

#计算拉普拉斯矩阵
def laplace_array(moon_x,moon_y,gamma):

    S = np.zeros((len(moon_x),len(moon_y)))

    for i in range(len(moon_x)):
        for j in range(len(moon_y)):
            S[i][j] = gaussian_2d(moon_x[i],moon_y[i],moon_x[j],moon_y[j],gamma)

    D = np.sum(S, axis=1)
    D = np.squeeze(np.array(D))
    D = np.diag(D)

    return D-S

#实现ratio_cut方法，返回归一化后的矩阵
def ratio_cut(laplace,k):

    val, vec = np.linalg.eig(laplace)
    id = np.argsort(val)
    topk_vecs = vec[:,id[0:k:1]]

    Sqrt = np.array(topk_vecs) * np.array(topk_vecs)

    divMat = np.tile(np.sqrt(np.transpose(sum(np.transpose(Sqrt)))), (2, 1))
    divMat = np.transpose(divMat)

    F = np.array(topk_vecs) / divMat

    return F


#kmeas算法
def kmeans(feature,class_num):

    core = []

    # 初始化
    for i in range(class_num):
        core.append(feature[i])

    flag = True

    while flag:

        K1 = np.zeros(feature.shape)
        K2 = np.zeros(feature.shape)
        count1 = 0
        count2 = 0

        for i in range(len(feature)):
            if sum((feature[i]-core[0])**2) <= sum((feature[i]-core[1])**2):
                K1[i] = feature[i]
                count1 += 1
            else:
                K2[i] = feature[i]
                count2 += 1

        sm1 = sum(K1) / count1
        sm2 = sum(K2) / count2

        count = 0

        if (sm1[0] == core[0][0]) & (sm1[1] == core[0][1]):
            count += 1
        else:
            core[0] = sm1#更新均值

        if (sm2[0] == core[1][0]) & (sm2[1] == core[1][1]):
            count += 1
        else:
            core[1] = sm2

        if count == 2:
            flag = False#分类完成，结束循环

    return K1,K2

#画图
def plot_moon(data1,data2):

    id_1 = []
    id_2 = []
    for i in range(len(data1)):
        if (data1[i][0] != 0) & (data1[i][1] != 0):
            id_1.append(i)
    for i in range(len(data2)):
        if (data2[i][0] != 0) & (data2[i][1] != 0):
            id_2.append(i)

    id_1 = np.array(id_1)
    id_2 = np.array(id_2)

    result1 = moon_dat[id_1]
    result2 = moon_dat[id_2]

    plt.figure()
    plt.scatter(result1[:, 0], result1[:, 1], color='b')
    plt.scatter(result2[:, 0], result2[:, 1], color='r')
    plt.show()

    return
if __name__ == '__main__':
    gamma = 10 ** 3
    L = laplace_array(moon_x, moon_y, gamma)
    F = ratio_cut(L, 2)
    K1, K2 = kmeans(F, 2)
    plot_moon(K1, K2)
