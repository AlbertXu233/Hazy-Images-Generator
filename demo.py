import matplotlib.pyplot as plt
import numpy as np

from dataset import HazyData
rootDir = 'demo'
alpha = 1
beta = 0.4
bias = 0.4
dataGenerator = HazyData(rootDir,alpha,beta,bias,random=False,extraAugment=False)
for i in range(6):
    hazy,origin,name = dataGenerator.__getitem__(i)
    hazy = np.transpose(hazy[0],[1,2,0])
    origin = np.transpose(origin[0],[1,2,0])
    plt.figure('hazy')
    plt.imshow(hazy)
    plt.figure('origin')
    plt.imshow(origin)
    plt.show()