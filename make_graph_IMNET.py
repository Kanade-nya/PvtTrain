'''
画图
dataset-IMAGENET
'''

import pandas as pd
import csv
import numpy as np
from matplotlib import pyplot as plt
import random as rd
import json

conv = []
with open(f'results/ori_shunted_with_conv_IMNET') as f:
    for index in f.readlines():
        # print(json.loads(index)["test_acc1"])
        conv.append(json.loads(index)["test_acc1"])
conv_x = np.linspace(0, len(conv) - 1, len(conv), dtype=int)

shunted = []
with open(f'results/shunted_imagenet') as f:
    for index in f.readlines():
        # print(json.loads(index)["test_acc1"])
        shunted.append(json.loads(index)["test_acc1"])
shunted_x = np.linspace(0, len(shunted) - 1, len(shunted), dtype=int)

# print(length)
fig, ax = plt.subplots()
#

ax.set_xticks([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600])
ax.set_yticks([0, 20, 40, 60, 80, 100])
#
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
#
plt.grid(color='gray', linestyle='-', linewidth='1')
#

pic = ax.plot(conv_x, conv, shunted_x, shunted)
ax.legend(pic, ['shunted_with_conv', 'shunted'])
plt.show()
