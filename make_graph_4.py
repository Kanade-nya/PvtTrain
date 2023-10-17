'''
画图
dataset-cifar100
'''

import pandas as pd
import csv
import numpy as np
from matplotlib import pyplot as plt
import random as rd

import json

re_shunted = []
with open('results/re_shunted') as f:
    for index in f.readlines():
        # print(json.loads(index)["test_acc1"])
        re_shunted.append(json.loads(index)["test_acc1"])

ori_shunted_with_conv = []
with open('results/ori_shunted_with_conv') as f:
    for index in f.readlines():
        # print(json.loads(index)["test_acc1"])
        ori_shunted_with_conv.append(json.loads(index)["test_acc1"])

ori_shunted_with_conv_2 = []
with open('results/ori_shunted_with_conv_2') as f:
    for index in f.readlines():
        # print(json.loads(index)["test_acc1"])
        ori_shunted_with_conv_2.append(json.loads(index)["test_acc1"])

ori_shunted_conv_outer = []
with open('results/ori_shunted_conv_outer') as f:
    for index in f.readlines():
        # print(json.loads(index)["test_acc1"])
        ori_shunted_conv_outer.append(json.loads(index)["test_acc1"])

ori_shunted = []
with open('results/ori_shunted') as f:
    for index in f.readlines():
        # print(json.loads(index)["test_acc1"])
        ori_shunted.append(json.loads(index)["test_acc1"])

re_shunted_conv = []
with open('results/re_shunted_conv') as f:
    for index in f.readlines():
        # print(json.loads(index)["test_acc1"])
        re_shunted_conv.append(json.loads(index)["test_acc1"])

ori_shunted_conv_outer_x = np.linspace(0, len(ori_shunted_conv_outer) - 1, len(ori_shunted_conv_outer), dtype=int)
re_shunted_x = np.linspace(0, len(re_shunted) - 1, len(re_shunted), dtype=int)
re_shunted_conv_x = np.linspace(0, len(re_shunted_conv) - 1, len(re_shunted_conv), dtype=int)
ori_shunted_with_conv_x = np.linspace(0, len(ori_shunted_with_conv) - 1, len(ori_shunted_with_conv), dtype=int)
ori_shunted_with_conv_2_x = np.linspace(0, len(ori_shunted_with_conv_2) - 1, len(ori_shunted_with_conv_2), dtype=int)
ori_shunted_x = np.linspace(0, len(ori_shunted) - 1, len(ori_shunted), dtype=int)
# print(length)
fig, ax = plt.subplots()
#

ax.set_xticks([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600])
ax.set_yticks([0, 20, 40, 60, 80,
               100])
#
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
#
plt.grid(color='gray', linestyle='-', linewidth='1')
#

pic = ax.plot(re_shunted_x, re_shunted, re_shunted_conv_x, re_shunted_conv,
              ori_shunted_with_conv_x, ori_shunted_with_conv, ori_shunted_x, ori_shunted, ori_shunted_conv_outer_x,
              ori_shunted_conv_outer, ori_shunted_with_conv_2_x, ori_shunted_with_conv_2)
ax.legend(pic, ['re_shunted', 're_shunted_conv', 'ori_shunted_with_conv', 'ori_shunted', 'ori_shunted_conv_outer','ori_shunted_with_conv_2'])
plt.show()
