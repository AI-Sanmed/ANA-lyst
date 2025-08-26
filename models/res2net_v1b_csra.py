#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from models import res2net_v1b
from models.csra import MHA
from models.dcd import DYCls


class MLP(torch.nn.Module):
    def __init__(self, dim, projection_size, hidden_size=4096):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_size),
            torch.nn.BatchNorm1d(hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)


class Res2NetV1bCsraMTML(res2net_v1b.Res2Net):
    def __init__(self, num_heads, lam, num_classes, input_dim=2048, pretrained=True):
        super(Res2NetV1bCsraMTML, self).__init__(res2net_v1b.Bottle2neck, (3, 4, 6, 3), baseWidth=26, scale=4)
        if pretrained:
            print("backbone params inited by ImageNet pre-trained model")
            model_url = 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth'
            self.load_state_dict(model_zoo.load_url(model_url, map_location=torch.device('cpu')))

        self.fc = nn.Sequential()

        self.num_classes = num_classes
        self.num_regressions = 1

        self.classifier = MHA(num_heads, lam, input_dim, num_classes)

        self.regressor = nn.ModuleList([torch.nn.Sequential(
            DYCls(input_dim, 896),
            MLP(896, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(inplace=True),
            nn.Linear(512, num_classes * self.num_regressions),
        )])

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        classification = self.classifier(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        regression = self.regressor[0](x)
        if self.num_regressions > 1:
            regression = regression.view(-1, self.num_classes, self.num_regressions)

        return classification, regression


class Res2NetV1bCsraMLC(Res2NetV1bCsraMTML):
    """multi-label classification"""
    def __init__(self, num_heads, lam, num_classes, input_dim=2048, pretrained=True):
        super(Res2NetV1bCsraMLC, self).__init__(
            num_heads, lam, num_classes, input_dim=input_dim,
            pretrained=pretrained)

        self.regressor = nn.ModuleList([nn.Linear(input_dim, self.num_classes * self.num_regressions)])

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        classification = self.classifier(x)

        return classification


if __name__ == '__main__':
    print()
