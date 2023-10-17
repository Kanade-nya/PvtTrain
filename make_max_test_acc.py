import json

max_acc = 0
shunted_imagenet = []
with open('results/shunted_cifar_300') as f:
    for index in f.readlines():
        # print(json.loads(index)["test_acc1"])
        acc1 = json.loads(index)["test_acc1"]
        shunted_imagenet.append(acc1)
        if acc1 >= max_acc:
            max_acc = acc1
print(max_acc)