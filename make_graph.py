import pandas as pd
import csv
import numpy as np
from matplotlib import pyplot as plt
import random as rd

import json

repvt_re =[]
with open('cifar100-300ep-repvt-2') as f:
    for index in f.readlines():
        # print(json.loads(index)["test_acc1"])
        repvt_re.append(json.loads(index)["test_acc1"])

result = []
with open('cifar100-300ep-convpvtmax') as f:
    for index in f.readlines():
        # print(json.loads(index)["test_acc1"])
        result.append(json.loads(index)["test_acc1"])

mlp_re = []

with open('cifar100-300ep-convpvt_mlp') as f:
    for index in f.readlines():
        # print(json.loads(index)["test_acc1"])
        mlp_re.append(json.loads(index)["test_acc1"])

inc_re = []

with open('inc_pvt') as f:
    for index in f.readlines():
        # print(json.loads(index)["test_acc1"])
        inc_re.append(json.loads(index)["test_acc1"])

inc_dw_1 = []
inc_dw_2 = []
with open('results/inc_dw_1') as f:
    for index in f.readlines():
        # print(json.loads(index)["test_acc1"])
        inc_dw_1.append(json.loads(index)["test_acc1"])

with open('inc_pvt') as f:
    for index in f.readlines():
        # print(json.loads(index)["test_acc1"])
        inc_dw_2.append(json.loads(index)["test_acc1"])

# print(result)
# print(len(result))
l1 = np.linspace(0,len(repvt_re)-1,len(repvt_re),dtype=int)
length = np.linspace(0,len(result)-1,len(result),dtype=int)
l2 = np.linspace(0,len(mlp_re)-1,len(mlp_re),dtype=int)
l3 = np.linspace(0,len(inc_re)-1,len(inc_re),dtype=int)

l_inc1 =np.linspace(0,len(inc_dw_1)-1,len(inc_dw_1),dtype=int)
l_inc2 = np.linspace(0,len(inc_dw_2)-1,len(inc_dw_2),dtype=int)
# print(length)
fig,ax = plt.subplots()
#

ax.set_xticks([0,50,100,150,200,250,300])
ax.set_yticks([0,20,40,60,80,100])
#
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
#
plt.grid(color='gray',linestyle='-',linewidth='1')
#
epochs = length
# print(len(epochs))
pic = ax.plot(l1,repvt_re,length,result,l2,mlp_re,l3,inc_re,l_inc1,inc_dw_1,l_inc2,inc_dw_2)
ax.legend(pic,['pvt_tiny','OUR','CONVPVT','incption','bingxing1','bingxing2'])
# plt.savefig("D:/plt")
plt.show()