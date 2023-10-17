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

pvt = []
with open('results/pvt') as f:
    for index in f.readlines():
        # print(json.loads(index)["test_acc1"])
        pvt.append(json.loads(index)["test_acc1"])

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

shunted_cifar = []
with open('results/shunted_cifar') as f:
    for index in f.readlines():
        # print(json.loads(index)["test_acc1"])
        shunted_cifar.append(json.loads(index)["test_acc1"])

shunted_cifar_300 = []
with open('results/shunted_cifar_300') as f:
    for index in f.readlines():
        # print(json.loads(index)["test_acc1"])
        shunted_cifar_300.append(json.loads(index)["test_acc1"])

shunted_cifar_600 = []
with open('results/shunted_cifar_600') as f:
    for index in f.readlines():
        # print(json.loads(index)["test_acc1"])
        shunted_cifar_600.append(json.loads(index)["test_acc1"])

inc_dw1 = []
with open('results/inc_dw_1') as f:
    for index in f.readlines():
        # print(json.loads(index)["test_acc1"])
        inc_dw1.append(json.loads(index)["test_acc1"])

inc_dw2 = []
with open('results/inc_dw_2') as f:
    for index in f.readlines():
        # print(json.loads(index)["test_acc1"])
        inc_dw2.append(json.loads(index)["test_acc1"])

inc_dw2_600 = []
with open('results/inc_dw_2_600') as f:
    for index in f.readlines():
        # print(json.loads(index)["test_acc1"])
        inc_dw2_600.append(json.loads(index)["test_acc1"])

convpvt = []
with open('results/conv_pvt') as f:
    for index in f.readlines():
        # print(json.loads(index)["test_acc1"])
        convpvt.append(json.loads(index)["test_acc1"])


ori_shunted_conv_outer_x = np.linspace(0, len(ori_shunted_conv_outer) - 1, len(ori_shunted_conv_outer), dtype=int)
shunted_cifar_x = np.linspace(0, len(shunted_cifar) - 1, len(shunted_cifar), dtype=int)
shunted_cifar_300_x = np.linspace(0, len(shunted_cifar_300) - 1, len(shunted_cifar_300), dtype=int)
shunted_cifar_600_x = np.linspace(0, len(shunted_cifar_600) - 1, len(shunted_cifar_600), dtype=int)
inc_dw1_x = np.linspace(0, len(inc_dw1) - 1, len(inc_dw1), dtype=int)
inc_dw2_x = np.linspace(0, len(inc_dw2) - 1, len(inc_dw2), dtype=int)
inc_dw2_600_x = np.linspace(0, len(inc_dw2_600) - 1, len(inc_dw2_600), dtype=int)
convpvt_x = np.linspace(0, len(convpvt) - 1, len(convpvt), dtype=int)
pvt_x = np.linspace(0, len(pvt) - 1, len(pvt), dtype=int)
re_shunted_x = np.linspace(0, len(re_shunted) - 1, len(re_shunted), dtype=int)
re_shunted_conv_x = np.linspace(0, len(re_shunted_conv) - 1, len(re_shunted_conv), dtype=int)
ori_shunted_with_conv_x = np.linspace(0, len(ori_shunted_with_conv) - 1, len(ori_shunted_with_conv), dtype=int)
ori_shunted_x = np.linspace(0, len(ori_shunted) - 1, len(ori_shunted), dtype=int)
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

pic = ax.plot(shunted_cifar_300_x, shunted_cifar_300, re_shunted_x, re_shunted, re_shunted_conv_x, re_shunted_conv,
              ori_shunted_with_conv_x, ori_shunted_with_conv,ori_shunted_x,ori_shunted,ori_shunted_conv_outer_x,ori_shunted_conv_outer)
ax.legend(pic, ['shunted-300', 're_shunted', 're_shunted_conv', 'ori_shunted_with_conv','ori_shunted','ori_shunted_conv_outer'])
plt.show()
