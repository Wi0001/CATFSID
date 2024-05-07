import numpy as np
import glob
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader

def data_for_dfs(root_dir):
    data_list = glob.glob(root_dir + '/data/*.npy')
    label_list = glob.glob(root_dir + '/label/*.npy')
    domain_list = glob.glob(root_dir + '/domain/*.npy')
    WiFi_data = {}
    for data_dir in data_list:
        print(data_dir)
        data_name = data_dir.split('/')[-1].split('.')[0]
        with open(data_dir, 'rb') as f:
            data = np.load(f)
            data = data.reshape(((len(data),12,1,26,1000)))
            #data = data[:,:, :, 0::5]
            data = np.swapaxes(np.swapaxes(data, 1, 4),3,4)
            print(data.shape)
            #data = np.expand_dims(data, axis=-3)
        WiFi_data[data_name] = torch.Tensor(data)
    for label_dir in label_list:
        label_name = label_dir.split('/')[-1].split('.')[0]
        with open(label_dir, 'rb') as f:
            #读的是表里的东西
            label = np.load(f)
        WiFi_data[label_name] = torch.Tensor(label)
    for domain_dir in domain_list:
        domain_name = domain_dir.split('/')[-1].split('.')[0]
        with open(domain_dir, 'rb') as f:
            domain = np.load(f)
        WiFi_data[domain_name] = torch.Tensor(domain)
    print('==================',WiFi_data)
    return WiFi_data

def data_for_csi(root_dir):
    data_list = glob.glob(root_dir + '/data/*.npy')
    label_list = glob.glob(root_dir + '/label/*.npy')
    domain_list = glob.glob(root_dir + '/domain/*.npy')
    WiFi_data = {}
    for data_dir in data_list:
        print(data_dir)
        data_name = data_dir.split('/')[-1].split('.')[0]
        with open(data_dir, 'rb') as f:
            data = np.load(f)
            data = data.reshape((len(data), 1, 1000, 180))
        WiFi_data[data_name] = torch.Tensor(data)
    for label_dir in label_list:
        label_name = label_dir.split('/')[-1].split('.')[0]
        with open(label_dir, 'rb') as f:
            label = np.load(f)
        WiFi_data[label_name] = torch.Tensor(label)
    for domain_dir in domain_list:
        domain_name = domain_dir.split('/')[-1].split('.')[0]
        with open(domain_dir, 'rb') as f:
            domain = np.load(f)
        WiFi_data[domain_name] = torch.Tensor(domain)
    #print('==================',WiFi_data)
    return WiFi_data

def data_for_ceshi(root_dir):
    data_list = glob.glob(root_dir + '/data/*.npy')
    label_list = glob.glob(root_dir + '/label/*.npy')
    WiFi_data = {}
    for data_dir in data_list:
        print(data_dir)
        data_name = data_dir.split('/')[-1].split('.')[0]
        with open(data_dir, 'rb') as f:
            data = np.load(f)
            data = data.reshape((len(data), 1, 250, 90))
        WiFi_data[data_name] = torch.Tensor(data)
    for label_dir in label_list:
        label_name = label_dir.split('/')[-1].split('.')[0]
        with open(label_dir, 'rb') as f:
            label = np.load(f)
        WiFi_data[label_name] = torch.Tensor(label)
    #print('==================',WiFi_data)
    return WiFi_data