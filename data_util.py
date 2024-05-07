import os
import numpy as np
#from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import pywt
from matplotlib import pyplot as plt
from pywt._doc_utils import wavedec2_keys, draw_2d_wp_basis
from sklearn.decomposition import PCA
import scipy.signal as signal
from scipy import interpolate
from scipy.interpolate import pchip_interpolate
from wifilib import *

#hampel滤波器
def hampel(X,k):
    length = X.shape[0] - 1
    nsigma = 3
    iLo = np.array([i - k for i in range(0, length + 1)])
    iHi = np.array([i + k for i in range(0, length + 1)])
    iLo[iLo < 0] = 0
    iHi[iHi > length] = length
    xmad = []
    xmedian = []
    for i in range(length + 1):
        w = X[iLo[i]:iHi[i] + 1]
        medj = np.median(w)
        mad = np.median(np.abs(w - medj))
        xmad.append(mad)
        xmedian.append(medj)
    xmad = np.array(xmad)
    xmedian = np.array(xmedian)
    scale = 1.4826  # 缩放
    xsigma = scale * xmad
    xi = ~(np.abs(X - xmedian) <= nsigma * xsigma)  # 找出离群点（即超过nsigma个标准差）
    # 将离群点替换为中为数值
    xf = X.copy()
    xf[xi] = xmedian[xi]
    return xf

#pca降维
def pcause(A):
    pca = PCA(n_components=12)
    pca.fit(A)
    return pca.transform(A)

#截取固定长度的csv文件
def data_len1(csi_np,length):

    len1 = len(csi_np)
    csi_np_list=list(csi_np)
    if len1<=length:
        d=length-len1
        csi_np_1=np.zeros((d,len(csi_np[1])))
        csi_np_1_list=list(csi_np_1)

        csi_np_list.extend(csi_np_1_list)

        csi_np2=np.array(csi_np_list)

    else:
        csi_np2 = csi_np[0:length, :]
    return csi_np2


def data_len(csi_amp, length):
    len1 = len(csi_amp)
    d = len1 - length
    sta = int(d)
    end = int(len1)

    csi_amp1 = csi_amp[sta:end, :, :, :]
    #csi_amp1 = csi_amp1[::4, :, :, :]
    return csi_amp1
#去除静态成分
def remove(x):
    return x