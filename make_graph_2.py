'''
dataset-imagenet
'''

import pandas as pd
import csv
import numpy as np
from matplotlib import pyplot as plt
import random as rd

import json



# inc_dw_1 = []
shunted_imagenet = []
# with open('results/inc_dw_1') as f:
#     for index in f.readlines():
#         # print(json.loads(index)["test_acc1"])
#         inc_dw_1.append(json.loads(index)["test_acc1"])

with open('results/shunted_imagenet') as f:
    for index in f.readlines():
        # print(json.loads(index)["test_acc1"])
        shunted_imagenet.append(json.loads(index)["test_acc1"])

# print(result)
# print(len(result))
# l1 = np.linspace(0,len(repvt_re)-1,len(repvt_re),dtype=int)
# length = np.linspace(0,len(result)-1,len(result),dtype=int)
# l2 = np.linspace(0,len(mlp_re)-1,len(mlp_re),dtype=int)
# l3 = np.linspace(0,len(inc_re)-1,len(inc_re),dtype=int)

# l_inc1 =np.linspace(0,len(inc_dw_1)-1,len(inc_dw_1),dtype=int)
s_image = np.linspace(0,len(shunted_imagenet)-1,len(shunted_imagenet),dtype=int)
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
# epochs = length
# print(len(epochs))
pic = ax.plot(s_image,shunted_imagenet)
ax.legend(pic,['shunted-vit'])
# plt.savefig("D:/plt")
plt.show()