import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import linalg
from wifilib import *
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from data_util import *
import scipy.signal as signal
from scipy.interpolate import pchip_interpolate
from sklearn import preprocessing
import csv
import glob
import os
import scipy
import torch
import pywt
from skimage.restoration import denoise_wavelet
from sklearn.utils.extmath import svd_flip
import shutil

def time_prune(bf):
    csi_list = list(map(get_scale_csi, bf))
    csi_np = (np.array(csi_list))
    csi_amp = np.abs(csi_np)
    sp1 = 0.005
    sp2 = "0.005"

    p = len(sp2) - str.find(sp2,'.') -1  # 采样周期小数点位数
    #print(len(csi_np))
    pack = len(csi_np)
    pack_before = len(csi_np)
    a = bf[0]["timestamp_low" ]   #将第一个元胞数组时间戳赋给a
    #for i in range(pack):
        #print("timestamp_low：", bf[i]["timestamp_low"])
        #print("i：", i)

    #exit(0)
    for i in range(pack):
        #print("timestamp_low：",bf[i]["timestamp_low"])
        #print("a：",a)
        b = (bf[i]["timestamp_low" ]-a) / 1000000
        #print("b：",b)
        c = np.round(b, p)   # 四舍五入到指定小数位
        bf[i]["timestamp_low" ] = c   #将四舍五入后时间戳赋回
        bf[i]["timestamp_low"] = np.round (np.ceil(c / sp1) * sp1,p ) # 时间戳取采样周期的整数倍 ceil取大于该数的最小整数
        #print("c：",c)
    for i in range(pack-1):
        j = i + 1
        if bf[j]["timestamp_low"] == bf[i]["timestamp_low"]:     # 相邻时间戳是否相等
            k = j
            while(k <= pack-1):
                bf[k]["timestamp_low"] = np.round((bf[k]["timestamp_low"]) + sp1, p)#相等时间戳后的所有时间戳加采样间隔
                k = k+ 1

    #开始插值
    i = 0
    timestamp = []#将CSI中的时间戳赋给新的行矩阵
    for i in range(pack):
            timestamp.append(bf[i]["timestamp_low"])
    timestamp = (np.array(timestamp)).T
    #print("timestamp", timestamp)
    pack = (timestamp[pack-1] - timestamp[0]) / sp1 + 1 #共有多少时间点要循环
    #print("pack",pack)
    #print("pack",pack_before)
    L = pack - pack_before
    P = L / pack * 100
    #print("pack_before：",pack_before,"   pack：",pack,"   丢包率：",np.round(P,2),"%")

    for i in range(int(pack - 1)):
        j = i + 1
        L = round((timestamp[j] - timestamp[i]) / sp1 - 1)   #相邻时间戳之间丢包个数
        if  L != 0:
            D = (csi_amp[j,:] - csi_amp[i,:]) / (L + 1)  # AM1的差值矩阵
            q = 0
            while( q!=L):
                a = csi_amp[i,:]+D * (q + 1)  #计算i后第一个AM1值

                b = np.round(timestamp[i]+sp1 * (q + 1),3) #计算i后第一个时间值
                p = len(timestamp)  #补值后矩阵长度

                timestamp1 = np.append( timestamp[0:i+q+1], b)
                timestamp2 = np.append(timestamp1 ,timestamp[j + q:p])  #进行补时间值
                timestamp =  timestamp2

                a = np.expand_dims(a,0)
                csi_amp1 = np.concatenate( [csi_amp[0:i+q+1], a],0)
                csi_amp2 = np.concatenate([csi_amp1,csi_amp [j + q:p]],0)   # 进行补AM1值
                csi_amp = csi_amp2
                q = q + 1
    return csi_amp


def M_csv(input_path,outpath):
    #读取路径下所有文件名
    for root, dirs, files in os.walk(input_path):
        print(root)  # 当前目录路径
        print(dirs)  # 当前路径下所有子目录
        print(files)  # 当前路径下所有非目录子文件

    for i in range(len(files)):
        #print(files[i])
        outname = files[i].split('.')[0]  # 得到无后缀名的文件名
        path = input_path + files[i]
        bf = read_bf_file(path)
        csi_list = list(map(get_scale_csi, bf))
        csi_np = (np.array(csi_list))
        csi_amp = np.abs(csi_np)
        # 进行插值
        csi_amp = time_prune(bf)
        #print(len(csi_amp))
        #移除背景环境值
        #将包截成相等长度
        csi_amp1 = data_len(csi_amp, 1000)
        #利用hampel滤波器和巴特沃斯低通滤波器
        [b, a] = signal.butter(6, 60 / (1000 / 2), 'lowpass')
        threshold = 0.3
        for j in range(csi_amp1.shape[1]):
            for k in range(csi_amp1.shape[2]):
                for m in range(csi_amp1.shape[3]):
                    csi_amp1[:, j, k, m] = hampel(csi_amp1[:, j, k, m], 3)
                    csi_amp1[:, j, k, m] = signal.filtfilt(b, a, csi_amp1[:, j, k, m])

                    w = pywt.Wavelet('sym8')

                    maxlev = pywt.dwt_max_level(len(csi_amp1[:, j, k, m]), w.dec_len)
                    coffs = pywt.wavedec(csi_amp1[:, j, k, m], 'sym8', level=maxlev)
                    for i in range(1, len(coffs)):
                        coffs[i] = pywt.threshold(coffs[i], threshold * max(coffs[i]))
                    csi_amp1[:, j, k, m] = pywt.waverec(coffs, 'sym8')

        data=np.zeros((len(csi_amp1), 90))
        for q in range(len(csi_amp1)):
            data1_1 = np.hstack((csi_amp1[q][0][0], csi_amp1[q][0][1]))
            data[q, :] = np.hstack((data1_1, csi_amp1[q][0][2]))
        data=pcause(data)
        #print(data.shape)
        zx1=np.zeros(((len(data[1]),26,1000)))
        #print(len(data[1]))
        window_size = round(200 / 4 + 1)
        for i in range(len(data[1])):
            f, t, zx = signal.stft(data[:,i], fs=200,nperseg=window_size,noverlap=window_size-1,nfft=window_size)
            #print(f.shape)
            #print(t.shape)
            #print(zx)
            #print(zx.shape)
            #plt.plot(zx)
            #plt.show()
            zx1[i,:,:]=zx
        #print(zx1.shape)
        '''
        data = np.zeros((len(csi_amp1), 90))
        # 随机选择一个接收天线
        for q in range(len(csi_amp1)):
            data1_1 = np.hstack((csi_amp1[q][0][0], csi_amp1[q][0][1]))
            data[q, :] = np.hstack((data1_1, csi_amp1[q][0][2]))
        '''
        zx1=zx1.reshape(12,26*1000)
        with open( outpath +outname + '.csv', 'w',newline='') as csvfile:
           #写到表格里的总数组
            writer = csv.writer(csvfile)
            for p in range(len(zx1)):
                writer.writerow(zx1[p,])
def train_csi_lable(inpath,outpath):
    for root, dirs, files in os.walk(inpath):
        print(root)  # 当前目录路径
        print(dirs)  # 当前路径下所有子目录
        print(files)  # 当前路径下所有非目录子文件
    dataall1 = np.zeros(913)
    dataall3 = np.zeros(913)
    label = ['pick','sit','stand', 'walk','circle','round','push']
    dataall2 = np.zeros((913,12,26000))
    a = 0
    for name in files:
        with open(inpath+name, encoding="utf8") as f:  # 打开csv文件
            csv_reader = csv.reader(f)
            data2 = np.zeros((12,26000))
            j = 0
            for line in csv_reader:
                data1 = np.zeros(26000)
                for i in range(26000):
                    #print(line[i])
                    data1[i] = np.array(line[i])
                    if data1[i] == np.nan:
                        print("错误啦")
                        exit()
                data2[j] = data1
                j += 1
                dataall2[a] = data2
        if name.find(label[0]) >= 0:
            dataall1[a] = 0
        if name.find(label[1]) >= 0:
            dataall1[a] = 1
        if name.find(label[2]) >= 0:
            dataall1[a] = 2
        if name.find(label[3]) >= 0:
            dataall1[a] = 3
        if name.find(label[4]) >= 0:
            dataall1[a] = 4
        if name.find(label[5]) >= 0:
            dataall1[a] = 5
        if name.find(label[6]) >= 0:
            dataall1[a] = 6

        if int(name.split('-')[1]) == 1:
            dataall3[a] = 0
        if int(name.split('-')[1]) == 2:
            dataall3[a] = 1
        if int(name.split('-')[1]) == 3:
            dataall3[a] = 2
        a = a + 1
    print(dataall2.shape)
    print(dataall2)
    print(dataall1)
    print(dataall3)
    #保存文件
    np.save(outpath+"data-train", dataall2)
    np.save(outpath+"label-train", dataall1)
    np.save(outpath +"domain-train", dataall3)
def val_csi_lable(inpath,outpath):
    for root, dirs, files in os.walk(inpath):
        print(root)  # 当前目录路径
        print(dirs)  # 当前路径下所有子目录
        print(files)  # 当前路径下所有非目录子文件
    dataall1 = np.zeros(913)
    dataall3 = np.zeros(913)
    label = ['pick','sit','stand', 'walk','circle','round','push']
    dataall2 = np.zeros((913,250,90))
    a = 0
    for name in files:
        with open(inpath+name, encoding="utf8") as f:
            csv_reader = csv.reader(f)
            data2 = np.zeros((250, 90))
            j = 0
            for line in csv_reader:
                data1 = np.zeros(90)
                for i in range(90):
                    data1[i] = np.array(line[i])
                data2[j] = data1
                j += 1
                dataall2[a] = data2
        if name.find(label[0]) >= 0:
            dataall1[a] = 0
        if name.find(label[1]) >= 0:
            dataall1[a] = 1
        if name.find(label[2]) >= 0:
            dataall1[a] = 2
        if name.find(label[3]) >= 0:
            dataall1[a] = 3
        if name.find(label[4]) >= 0:
            dataall1[a] = 4
        if name.find(label[5]) >= 0:
            dataall1[a] = 5
        if name.find(label[6]) >= 0:
            dataall1[a] = 6

        if int(name.split('-')[1]) == 1:
            dataall3[a] = 0
        if int(name.split('-')[1]) == 2:
            dataall3[a] = 1
        if int(name.split('-')[1]) == 3:
            dataall3[a] = 2
        a = a + 1
    print(dataall2.shape)
    print(dataall2)
    print(dataall1)
    print(dataall3)

    #保存文件
    np.save(outpath+"data-val", dataall2)
    np.save(outpath+"label-val", dataall1)
    np.save(outpath +"domain-val", dataall3)
def test_csi_lable(inpath,outpath):
    for root, dirs, files in os.walk(inpath):
        print(root)  # 当前目录路径
        print(dirs)  # 当前路径下所有子目录
        print(files)  # 当前路径下所有非目录子文件
    dataall1 = np.zeros(278)
    dataall3 = np.zeros(278)
    label = ['pick','sit','stand', 'walk','circle','round','push']
    dataall2 = np.zeros((278,12,26000))
    a = 0
    for name in files:
        with open(inpath+name, encoding="utf8") as f:
            csv_reader = csv.reader(f)
            data2 = np.zeros((12,26000))
            j = 0
            for line in csv_reader:
                data1 = np.zeros(26000)
                for i in range(26000):
                    data1[i] = np.array(line[i])
                data2[j] = data1
                j += 1
                dataall2[a] = data2
        if name.find(label[0]) >= 0:
            dataall1[a] = 0
        if name.find(label[1]) >= 0:
            dataall1[a] = 1
        if name.find(label[2]) >= 0:
            dataall1[a] = 2
        if name.find(label[3]) >= 0:
            dataall1[a] = 3
        if name.find(label[4]) >= 0:
            dataall1[a] = 4
        if name.find(label[5]) >= 0:
            dataall1[a] = 5
        if name.find(label[6]) >= 0:
            dataall1[a] = 6

        if int(name.split('-')[1]) == 1:

            dataall3[a] = 0
        if int(name.split('-')[1]) == 2:

            dataall3[a] = 1
        if int(name.split('-')[1]) == 3:

            dataall3[a] = 2
        a = a + 1
    print(dataall2.shape)
    print(dataall2)
    print(dataall1)
    print(dataall3)

    #保存文件
    np.save(outpath+"data-test", dataall2)
    np.save(outpath+"label-test", dataall1)
    np.save(outpath +"domain-test", dataall3)
def train_csi_lable1(inpath,outpath):
    for root, dirs, files in os.walk(inpath):
        print(root)  # 当前目录路径
        print(dirs)  # 当前路径下所有子目录
        print(files)  # 当前路径下所有非目录子文件
    dataall1 = np.zeros(20)
    dataall3 = np.zeros(20)
    label = ['pick','sit','stand', 'walk','circle','round','push']
    dataall2 = np.zeros((20,250,90))
    a = 0
    for name in files:
        with open(inpath+name, encoding="utf8") as f:  # 打开csv文件
            csv_reader = csv.reader(f)
            data2 = np.zeros((250, 90))  # 创建某一行数据数组 1200 *180 数据包*子载波数
            j = 0
            for line in csv_reader:
                data1 = np.zeros(90)
                for i in range(90):
                    data1[i] = np.array(line[i])
                    if data1[i] == np.nan:
                        print("错误啦")
                        exit()
                data2[j] = data1
                j += 1
                dataall2[a] = data2


        if int(name.split('-')[1]) == 1:
            dataall3[a] = 0
        if int(name.split('-')[1]) == 2:
            dataall3[a] = 1
        if int(name.split('-')[1]) == 3:
            dataall3[a] = 2
        a = a + 1
    print(dataall2.shape)
    print(dataall2)
    print(dataall3)
    #保存文件
    np.save(outpath+"data-train", dataall2)
    np.save(outpath +"label-train", dataall3)

#M_csv("D:/360Downloads/cy/my-frame/dataset_in_school/416/zyp/","D:/360Downloads/cy/my-frame/dataset_in_school/416/csv/")

test_csi_lable("D:/360Downloads/cy/my-frame/dataset_in_school/test/","D:/360Downloads/cy/my-frame/dataset_in_school/data_stft/")