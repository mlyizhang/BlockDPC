#! /usr/bin/env python
# -*- coding: utf-8 -*-

#############################
# utils files.
#############################
from numpy import *
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import json
import pickle
import math
from utils import *
import os
from sklearn.cluster import DBSCAN
import random
import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score
from typing import List, Tuple
from numpy import arange, argsort, argwhere, empty, full, inf, intersect1d, max, ndarray, sort, sum, zeros
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import cdist
import umap

def load_dataset(filepath):
    """
        Return:
            dataset: dict
    """
    with open(filepath, 'rb') as fr:
        dataset = pickle.load(fr)
    return dataset
# 按行的方式计算两个坐标点之间的距离
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)



def results(data_path):
    parameters=[]# number of centers in clients, number of centers in servers, k in SNN
    datapkl = load_dataset(data_path)
    #order = datapkl['order']
    data = list(datapkl['full_data'])
    corepoints = [] # save local kmeans centers.
    for i_client in range(datapkl['num_clusters']):
        lodata = datapkl["client_" + str(i_client)]
        noise = np.random.random((lodata.shape[0], lodata.shape[1]))
        # print(noise)
        array_0_1 = np.full((lodata.shape[0], lodata.shape[1]), 0.1)
        noise = noise.reshape(lodata.shape[0], lodata.shape[1])

        lodata = lodata * noise*array_0_1
        #计算每个局部数据点的反向k近邻
        if 'data_PenDigits2d' in data_path or 'mnist2d' in data_path:
            right=200
        else:
            right=150
        right=300
        n_clusters = min([len(lodata) // 2, right])
        # print('客户端kmeans中的k值',len(lodata)-5)
        cluster = KMeans(n_clusters).fit(lodata)
        # print('center',cluster.cluster_centers_)
        localcenter=cluster.cluster_centers_
        # noise = np.random.laplace(0, 0.5, localcenter.shape[0] * localcenter.shape[1])
        # noise = noise.reshape(localcenter.shape[0], localcenter.shape[1])
        # localcenter = localcenter + noise

        corepoints.append(localcenter)

    # server: process the information from clients
    serverdata = np.concatenate(corepoints, axis=0)
    #serverdata =np.array(corepoints)
    label = datapkl['true_label']
    parameters.append(n_clusters)
    cnum=len(set(label))
    parameters.append(cnum)
    #print('snndpc中的参数k',k )
    #centroid= SNN(k, cnum, serverdata)
    if 'data_PenDigits2d' in data_path or 'mnist2d' in data_path or 'usps2d' in data_path:
        k = 23
    else:
        k = 22
    k=25
    parameters.append(k)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(serverdata)
    distances, indices = nbrs.kneighbors(serverdata)
    rnn = []  # 每个数据点的反向最近邻
    for i in range(serverdata.shape[0]):
        templist = []
        for j in range(len(indices)):
            if i in indices[j][1:]:
                templist.append(j)
        rnn.append(templist)
    # 计算每个数据点的局部密度locald
    locald = []
    distances = cdist(serverdata, serverdata)
    for i in range(len(rnn)):
        a = len(rnn[i]) ** 2
        b = sum(distances[i, rnn[i]])
        if a == 0 and b == 0:
            a = 0
            b = 1
        locald.append(a / b)
    #求每个数据点的MNN
    mnn=[]
    for i in range(serverdata.shape[0]):
        templist=[]
        for j in range(len(indices)):
            if i!=j and i in indices[j] and j in indices[i]:
                templist.append(j)
        mnn.append(templist)
    #再求局部密度
    finallocald=[]
    for i in range(len(mnn)):
        a=len(mnn[i])
        locald=np.array(locald)
        b=sum(locald[mnn[i]])
        if a == 0 and b == 0:
            a = 0
            b = 1
        finallocald.append(a/b)

    sorted_list = sorted(locald, reverse=True)
    new_indices = [sorted_list.index(x) for x in locald]
    selected_index = new_indices[:cnum]
    #
    finalcenter=[]
    for i in selected_index:
        finalcenter.append(serverdata[i])
    idx = []
    # finalcenter=np.array(finalcenter)
    # noise = np.random.laplace(0, 0.5, finalcenter.shape[0] * finalcenter.shape[1])
    # noise = noise.reshape(finalcenter.shape[0], finalcenter.shape[1])
    # finalcenter = finalcenter + noise
    # print(finalcenter)
    # finlal centers +扰动
    # finalcenter.tolist()

    for i in data:
        simi = []
        for j in finalcenter:
            simi.append(np.linalg.norm(i - j))
        idx.append(simi.index(min(simi)) + 1)
    arr = np.array(idx)
    ari = round(adjusted_rand_score(label, arr),4 )
    nmi = round(normalized_mutual_info_score(label, arr),4)
    ami=  round(adjusted_mutual_info_score(label, arr),4)
    # print(ari,nmi)
    return ari, nmi,ami,parameters






