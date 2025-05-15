import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, actv=nn.ReLU):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.actv1 = actv()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.actv2 = actv()
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                              padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                               padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.actv1(self.bn1(x))
        else:
            out = self.actv1(self.bn1(x))
        out = self.actv2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, actv=nn.ReLU):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, actv)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, actv):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, actv))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class WideResNet40_4(nn.Module):
    def __init__(self, depth=40, widen_factor=4, num_classes=10, dropRate=0.0, in_channel=3, actv=nn.ReLU):
        super(WideResNet40_4, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(in_channel, nChannels[0], kernel_size=3, stride=1,
                              padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, actv)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, actv)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, actv)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.actv = actv()
        self.fc = nn.Linear(nChannels[3], num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.actv(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.fc.in_features)
        return self.fc(out)

def wideresnet40_4(in_channel=3, actv=nn.ReLU, **kwargs):
    return WideResNet40_4(in_channel=in_channel, actv=actv, **kwargs)