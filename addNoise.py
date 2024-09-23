import os.path

import pandas as pd,numpy as np,random

def translabel():
    traindata = pd.read_csv('TCUCN/data/train_chineselabel.txt',sep=None)
    valdata = pd.read_csv('TCUCN/data/val_chineselabel.txt', sep=None)
    testdata = pd.read_csv('TCUCN/data/test_chineselabel.txt', sep=None)
    label = traindata['label']
    testlabel=testdata['label']
    vallabel = valdata['label']
    label1 = np.asarray(label)
    label_list = np.unique(label1)
    # np.savetxt('TCUCN/data/labellist.csv',label_list)
    label_map = {}

    for (i, label2) in enumerate(label_list):
        label_map[label2] = i

    trainlabel=[]
    for j in label:
        maplabel = label_map[j]
        trainlabel.append(maplabel)
    traindata['label']=trainlabel
    traindata.to_csv('TCUCN/data/train.csv', index=0, sep='\t')

    testlabel_map = []
    for k in testlabel:
        maplabel1 = label_map[k]
        testlabel_map.append(maplabel1)
    testdata['label']=testlabel_map
    testdata.to_csv('TCUCN/data/test.csv', index=0, sep='\t')

    vallabel_map = []
    for l in vallabel:
        maplabel2 = label_map[l]
        vallabel_map.append(maplabel2)
    valdata['label'] = vallabel_map
    valdata.to_csv('TCUCN/data/dev.csv', index=0, sep='\t')

    print('ok')

def addNoise(noiserate,noisemode):
    # trans = {1: 2, 0: 0, 2: 3, 6:7, 8: 8, 9:8, 4: 4, 5: 4, 7: 7, 3:4}
    # trans = {0:0, 1:2, 2:3, 3:4, 4:1}
    trans = {0:1,1:0}
    content = pd.read_csv('AG/train.csv', sep='\t',engine='python')
    data = content['data']
    l = content['label']
    npl = np.array(l)
    ul = np.unique(npl)

    # path = os.path.join('SST2',str(noiserate))
    path = 'AG'
    dataname='train_' + str(noiserate) + '_' + noisemode + '.csv'

    num_noise = int(noiserate * len(data))
    noise_idx = list(np.random.randint(len(data),size=num_noise))
    # noise_idx_name = 'noise_idx' + '_' + noisemode + '.csv'
    # np.savetxt(os.path.join(path,noise_idx_name),noise_idx)

    if noisemode == 'sym':
        for i in noise_idx:
            labelidx = random.randint(0,1)
            noise_label = ul[labelidx]
            l[i] = noise_label
        content['label']=l
        content.to_csv(os.path.join(path,dataname),index=0, sep='\t')
    else:
        for i in noise_idx:
            noise_label = trans[l[i]]
            l[i] = noise_label
        content['label']=l
        content.to_csv(os.path.join(path,dataname),index=0, sep='\t')

def addNoise_new(noiserate,noisemode):
    # trans = {1: 2, 0: 0, 2: 3, 6:7, 8: 8, 9:8, 4: 4, 5: 4, 7: 7, 3:4}
    # trans = {0:0, 1:2, 2:3, 3:4, 4:1}
    trans = {0:1,1:0}
    content = pd.read_csv('AG/train.csv', sep='\t',engine='python')
    data = content['data']
    l = content['label']
    npl = np.array(l)
    ul = np.unique(npl)

    # path = os.path.join('SST2',str(noiserate))
    path = 'AG'
    dataname='train_' + str(noiserate) + '_' + noisemode + '.csv'

    if noisemode == 'sym':
        num_noise = int(noiserate * len(data))
        noise_idx = list(np.random.randint(len(data), size=num_noise))
        for i in noise_idx:
            labelidx = random.randint(0,1)
            noise_label = ul[labelidx]
            l[i] = noise_label
        content['label']=l
        content.to_csv(os.path.join(path,dataname),index=0, sep='\t')
    else:
        for i in noise_idx:
            num_noise = int(noiserate/2 * len(data))
            noise_label = trans[l[i]]
            l[i] = noise_label
        content['label']=l
        content.to_csv(os.path.join(path,dataname),index=0, sep='\t')

if __name__ == '__main__':
    noiserate = [0.2, 0.4, 0.6, 0.8]
    noisemode = ['sym', 'asym']
    for i in noiserate:
        for j in noisemode:
            addNoise(i,j)
