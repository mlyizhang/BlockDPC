from utils import *
import os
from sklearn.cluster import KMeans,DBSCAN
import matplotlib.pyplot as plt
import random
import numpy
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
import time
import datetime
current_time = datetime.datetime.now()
dataname=['abalonefed','Aggregationfed','coil20fed','data_PenDigitsfed','Gesturefed','glassfed',
          'heartfed','jainfed','leaffed','liverfed','oliver400featurefed','Pathbasedfed',
          'R15fed','spambasefed','umistfed','uspsfed','vehiclefed','waveformfed','Yeastfed']

dataname=['iris','breast','ecoli','zoo','thyroid','wine','seeds','abalone','heart','waveform',
          'gesture','liver','ionosphere']
dataname=['data_PenDigits2d','Covertype2d','mnist2d','usps2d']
dataname=['mnist2d']

for j in range(5):
    for i in dataname:
        start_time = time.time()
        data_path = '../dataset/'+i+'fed.pkl'
        datapkl = load_dataset(data_path)  # dataset is a json file
        print("Processing dataset:", data_path)
        print(datapkl['full_data'].shape)
        # print(datapkl['full_data'])
        a=[]
        n=[]
        am=[]
        for iter in range(200):
            ari, nmi,ami,parameters=results(data_path)
            a.append(ari)
            n.append(nmi)
            am.append(ami)

        print('ari',max(a),'nmi',max(n),'ami',max(am))
        end_time = time.time()
        print('number of centers in clients',parameters[0],'number of centers in servers',parameters[1],'k in knn',parameters[2])
        print("运行时间：", end_time - start_time, "秒")
        with open('result.txt','a') as f:
            f.write(data_path)
            f.write('ari'+str(max(a))+'nmi'+str(max(n))+'\n')
            f.write('number of centers in clients'+str(parameters[0])+'number of centers in servers'+
                    str(parameters[1])+'k in knn'+str(parameters[2])+'\n')
