from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import sys


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.stride=stride
        self.padding=padding
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False) # verify bias false
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001, # value found in tensorflow
                                 momentum=0.1, # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x, weights=None, name=None):
        if weights == None:
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
        else:
            x = F.conv2d(x, weights['{}.conv.weight'.format(name)],stride=self.stride, padding=self.padding)
            x = F.batch_norm(x, self.bn.running_mean, self.bn.running_var, weights['{}.bn.weight'.format(name)], weights['{}.bn.bias'.format(name)],training=True)            
            x = F.threshold(x, 0, 0, inplace=True)
        return x


class Mixed_5b(nn.Module):

    def __init__(self):
        super(Mixed_5b, self).__init__()

        self.conv0 = BasicConv2d(192, 96, kernel_size=1, stride=1)
        self.conv1 = BasicConv2d(192, 48, kernel_size=1, stride=1)
        self.conv2 = BasicConv2d(48, 64, kernel_size=5, stride=1, padding=2)
        self.conv3 = BasicConv2d(192, 64, kernel_size=1, stride=1)
        self.conv4 = BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1)
        self.conv5 = BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1)
        self.conv6 = BasicConv2d(192, 64, kernel_size=1, stride=1)

        self.branch0 = self.conv0

        self.branch1 = nn.Sequential(
            self.conv1, self.conv2
        )

        self.branch2 = nn.Sequential(
            self.conv3, self.conv4, self.conv5
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            self.conv6
        )

    def forward(self, x, weights=None):
        if weights == None:
            x0 = self.branch0(x)
            x1 = self.branch1(x)
            x2 = self.branch2(x)
            x3 = self.branch3(x)
        else:
            x0 = self.conv0(x,weights,'mixed_5b.conv0')
            x1 = self.conv1(x,weights,'mixed_5b.conv1')
            x1 = self.conv2(x1,weights,'mixed_5b.conv2')
            x2 = self.conv3(x,weights,'mixed_5b.conv3')
            x2 = self.conv4(x2,weights,'mixed_5b.conv4')
            x2 = self.conv5(x2,weights,'mixed_5b.conv5')
            x3 = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
            x3 = self.conv6(x3,weights,'mixed_5b.conv6')
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block35(nn.Module):

    def __init__(self, scale=1.0):
        super(Block35, self).__init__()

        self.scale = scale

        self.conv0 = BasicConv2d(320, 32, kernel_size=1, stride=1)
        self.conv1 = BasicConv2d(320, 32, kernel_size=1, stride=1)
        self.conv2 = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = BasicConv2d(320, 32, kernel_size=1, stride=1)
        self.conv4 = BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1)
        self.conv5 = BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1)

        self.branch0 = self.conv0

        self.branch1 = nn.Sequential(
            self.conv1, self.conv2
        )

        self.branch2 = nn.Sequential(
            self.conv3, self.conv4, self.conv5
        )

        self.conv2d = nn.Conv2d(128, 320, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x, weights=None, name=None):
        if weights==None:
            x0 = self.branch0(x)
            x1 = self.branch1(x)
            x2 = self.branch2(x)
            out = torch.cat((x0, x1, x2), 1)
            out = self.conv2d(out)
            out = out * self.scale + x
            out = self.relu(out)
        else:
            x0 = self.conv0(x,weights,'{}.conv0'.format(name))
            x1 = self.conv1(x,weights,'{}.conv1'.format(name))
            x1 = self.conv2(x1,weights,'{}.conv2'.format(name))
            x2 = self.conv3(x,weights,'{}.conv3'.format(name))
            x2 = self.conv4(x2,weights,'{}.conv4'.format(name))
            x2 = self.conv5(x2,weights,'{}.conv5'.format(name))
            out = torch.cat((x0, x1, x2), 1)
            out = F.conv2d(out, weights['{}.conv2d.weight'.format(name)], weights['{}.conv2d.bias'.format(name)], stride=1)
            out = out * self.scale + x
            out = F.threshold(out, 0, 0, inplace=True)
        return out


class Mixed_6a(nn.Module):

    def __init__(self):
        super(Mixed_6a, self).__init__()

        self.conv0 = BasicConv2d(320, 384, kernel_size=3, stride=2)
        self.conv1 = BasicConv2d(320, 256, kernel_size=1, stride=1)
        self.conv2 = BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = BasicConv2d(256, 384, kernel_size=3, stride=2)

        self.branch0 = self.conv0

        self.branch1 = nn.Sequential(
            self.conv1, self.conv2, self.conv3
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x, weights = None):
        if weights == None:
            x0 = self.branch0(x)
            x1 = self.branch1(x)
            x2 = self.branch2(x)
        else:
            x0 = self.conv0(x,weights,'mixed_6a.conv0')
            x1 = self.conv1(x,weights,'mixed_6a.conv1')
            x1 = self.conv2(x1,weights,'mixed_6a.conv2')
            x1 = self.conv3(x1,weights,'mixed_6a.conv3')
            x2 = F.max_pool2d(x, kernel_size=3, stride=2)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Block17(nn.Module):

    def __init__(self, scale=1.0):
        super(Block17, self).__init__()

        self.scale = scale

        self.conv0 = BasicConv2d(1088, 192, kernel_size=1, stride=1)
        self.conv1 = BasicConv2d(1088, 128, kernel_size=1, stride=1)
        self.conv2 = BasicConv2d(128, 160, kernel_size=(1,7), stride=1, padding=(0,3))
        self.conv3 = BasicConv2d(160, 192, kernel_size=(7,1), stride=1, padding=(3,0))

        self.branch0 = self.conv0

        self.branch1 = nn.Sequential(
            self.conv1, self.conv2, self.conv3
        )

        self.conv2d = nn.Conv2d(384, 1088, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x, weights=None, name=None):
        if weights==None:
            x0 = self.branch0(x)
            x1 = self.branch1(x)
            out = torch.cat((x0, x1), 1)
            out = self.conv2d(out)
            out = out * self.scale + x
            out = self.relu(out)
        else:
            x0 = self.conv0(x,weights,'{}.conv0'.format(name))
            x1 = self.conv1(x,weights,'{}.conv1'.format(name))
            x1 = self.conv2(x1,weights,'{}.conv2'.format(name))
            x1 = self.conv3(x1,weights,'{}.conv3'.format(name))
            out = torch.cat((x0, x1), 1)
            out = F.conv2d(out, weights['{}.conv2d.weight'.format(name)], weights['{}.conv2d.bias'.format(name)], stride=1)
            out = out * self.scale + x
            out = F.threshold(out, 0, 0, inplace=True)
        return out


class Mixed_7a(nn.Module):

    def __init__(self):
        super(Mixed_7a, self).__init__()

        self.conv0 = BasicConv2d(1088, 256, kernel_size=1, stride=1)
        self.conv1 = BasicConv2d(256, 384, kernel_size=3, stride=2)
        self.conv2 = BasicConv2d(1088, 256, kernel_size=1, stride=1)
        self.conv3 = BasicConv2d(256, 288, kernel_size=3, stride=2)

        self.conv4 = BasicConv2d(1088, 256, kernel_size=1, stride=1)
        self.conv5 = BasicConv2d(256, 288, kernel_size=3, stride=1, padding=1)
        self.conv6 = BasicConv2d(288, 320, kernel_size=3, stride=2)

        self.branch0 = nn.Sequential(
            self.conv0, self.conv1
        )

        self.branch1 = nn.Sequential(
            self.conv2, self.conv3
        )

        self.branch2 = nn.Sequential(
            self.conv4, self.conv5, self.conv6 
        )

        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x, weights = None):
        if weights == None:
            x0 = self.branch0(x)
            x1 = self.branch1(x)
            x2 = self.branch2(x)
            x3 = self.branch3(x)
        else:
            x0 = self.conv0(x,weights,'mixed_7a.conv0')
            x0 = self.conv1(x0,weights,'mixed_7a.conv1')
            x1 = self.conv2(x,weights,'mixed_7a.conv2')
            x1 = self.conv3(x1,weights,'mixed_7a.conv3')
            x2 = self.conv4(x,weights,'mixed_7a.conv4')
            x2 = self.conv5(x2,weights,'mixed_7a.conv5')
            x2 = self.conv6(x2,weights,'mixed_7a.conv6')
            x3 = F.max_pool2d(x, kernel_size=3, stride=2)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block8(nn.Module):

    def __init__(self, scale=1.0, noReLU=False):
        super(Block8, self).__init__()

        self.scale = scale
        self.noReLU = noReLU

        self.conv0 = BasicConv2d(2080, 192, kernel_size=1, stride=1)
        self.conv1 = BasicConv2d(2080, 192, kernel_size=1, stride=1)
        self.conv2 = BasicConv2d(192, 224, kernel_size=(1,3), stride=1, padding=(0,1))
        self.conv3 = BasicConv2d(224, 256, kernel_size=(3,1), stride=1, padding=(1,0))

        self.branch0 = self.conv0

        self.branch1 = nn.Sequential(
            self.conv1, self.conv2, self.conv3
        )

        self.conv2d = nn.Conv2d(448, 2080, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x, weights=None, name=None):
        if weights==None:
            x0 = self.branch0(x)
            x1 = self.branch1(x)
            out = torch.cat((x0, x1), 1)
            out = self.conv2d(out)
            out = out * self.scale + x
            if not self.noReLU:
                out = self.relu(out)
        else:
            x0 = self.conv0(x,weights,'{}.conv0'.format(name))
            x1 = self.conv1(x,weights,'{}.conv1'.format(name))
            x1 = self.conv2(x1,weights,'{}.conv2'.format(name))
            x1 = self.conv3(x1,weights,'{}.conv3'.format(name))
            out = torch.cat((x0, x1), 1)
            out = F.conv2d(out, weights['{}.conv2d.weight'.format(name)], weights['{}.conv2d.bias'.format(name)], stride=1)
            out = out * self.scale + x
            if not self.noReLU:
                out = F.threshold(out, 0, 0, inplace=True)
        return out


class InceptionResNetV2(nn.Module):

    def __init__(self, num_classes=1001):
        super(InceptionResNetV2, self).__init__()
        # Special attributs
        self.input_space = None
        self.input_size = (299, 299, 3)
        self.mean = None
        self.std = None
        # Modules
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.maxpool_5a = nn.MaxPool2d(3, stride=2)
        self.mixed_5b = Mixed_5b()
        self.repeat = nn.Sequential(
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17)
        )
        self.mixed_6a = Mixed_6a()
        self.repeat_1 = nn.Sequential(
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10)
        )
        self.mixed_7a = Mixed_7a()
        self.repeat_2 = nn.Sequential(
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20)
        )
        self.block8 = Block8(noReLU=True)
        self.conv2d_7b = BasicConv2d(2080, 1536, kernel_size=1, stride=1)
        self.avgpool_1a = nn.AvgPool2d(8, count_include_pad=False)
        self.last_linear = nn.Linear(1536, num_classes)

        self.block35 = Block35(scale=0.17)
        self.block17 = Block17(scale=0.10)
        self.block8 = Block8(scale=0.20)

    def forward(self, input, weights=None, get_feat=None):
        x = self.conv2d_1a(input, weights, 'conv2d_1a')
        x = self.conv2d_2a(x, weights, 'conv2d_2a')
        x = self.conv2d_2b(x, weights, 'conv2d_2b')
        if weights == None:
            x = self.maxpool_3a(x)
        else:
            x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.conv2d_3b(x, weights, 'conv2d_3b')
        x = self.conv2d_4a(x, weights, 'conv2d_4a')
        if weights == None:
            x = self.maxpool_5a(x)
        else:
            x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.mixed_5b(x, weights)
        if weights == None:
            x = self.repeat(x)
        else:
            for i in range(10):
                self.block35(x,weights, 'repeat.{}'.format(i))
        x = self.mixed_6a(x, weights)
        if weights == None:
            x = self.repeat_1(x)
        else:
            for i in range(20):
                self.block17(x,weights, 'repeat_1.{}'.format(i))
        x = self.mixed_7a(x, weights)
        if weights == None:
            x = self.repeat_2(x)
        else:
            for i in range(9):
                self.block8(x,weights, 'repeat_2.{}'.format(i))
        x = self.block8(x, weights, 'block8')
        x = self.conv2d_7b(x, weights, 'conv2d_7b')
        if weights == None:
            x = self.avgpool_1a(x)
        else:
            x = F.avg_pool2d(x, kernel_size=8, count_include_pad=False)
        features = x.view(x.size(0), -1)
        if weights == None:
            x = self.last_linear(features)
        else:
            x = F.linear(features, weights['last_linear.weight'], weights['last_linear.bias'])      
        if get_feat:
            return x,features
        else:
            return x


