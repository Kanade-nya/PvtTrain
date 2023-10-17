import torch
from torch import nn
import torch.nn.functional as F
from my_pvt.ori_shunted_conv_outer import ori_shunted_conv_outer
from my_pvt.ori_shunted_with_pconv import ori_shunted_with_pconv
from pvt_v2 import pvt_v2_b0
from thop import profile
class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        self.stem = nn.Sequential(*[
            Conv(3, 64, kernel_size=7, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])
        self.stages = nn.Sequential(*[
            self._make_stage(64, 256, down_sample=False, num_blocks=3),
            self._make_stage(256, 512, down_sample=True, num_blocks=4),
            self._make_stage(512, 1024, down_sample=True, num_blocks=6),
            self._make_stage(1024, 2048, down_sample=True, num_blocks=3),
        ])
        self.head = nn.Sequential(*[
            nn.AvgPool2d(kernel_size=7, stride=1, padding=0),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(2048, num_classes)
        ])

    @staticmethod
    def _make_stage(in_channels, out_channels, down_sample, num_blocks):
        layers = [Bottleneck(in_channels, out_channels, down_sample=down_sample)]
        for _ in range(1, num_blocks):
            layers.append(Bottleneck(out_channels, out_channels, down_sample=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.head(self.stages(self.stem(x)))





class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, down_sample=False, groups=1):
        super(Bottleneck, self).__init__()
        stride = 2 if down_sample else 1
        mid_channels = out_channels // 4
        self.shortcut = Conv(in_channels, out_channels, kernel_size=1, stride=stride, activation=False) \
            if in_channels != out_channels else nn.Identity()
        self.conv = nn.Sequential(*[
            Conv(in_channels, mid_channels, kernel_size=1, stride=1),
            Conv(mid_channels, mid_channels, kernel_size=3, stride=stride, groups=groups),
            Conv(mid_channels, out_channels, kernel_size=1, stride=1, activation=False)
        ])

    def forward(self, x):
        y = self.conv(x) + self.shortcut(x)
        return F.relu(y, inplace=True)


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, groups=1, activation=True):
        super(Conv, self).__init__()
        padding = kernel_size // 2 if padding is None else padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


if __name__ == '__main__':
    net = ori_shunted_with_pconv()
    print(net)
    print('Number of network parameters:', sum(param.numel() for param in net.parameters()))
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))
    # params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # print('Number: ',params)
    input = torch.randn(1, 3, 224, 224)
    flops, params = profile(net, inputs=(input,))
    print(flops)
    print(params)