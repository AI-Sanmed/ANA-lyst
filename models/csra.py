#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from models.dcd import conv_dy


class DynamicCSRA(nn.Module):
    def __init__(self, input_dim, num_classes, T, lam):
        super(DynamicCSRA, self).__init__()
        self.T = T  # temperature
        self.lam = lam  # Lambda
        self.head = conv_dy(input_dim, num_classes, 1, 1, 0)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        score = self.head(x) / torch.norm(self.head.p.weight, dim=1, keepdim=True).transpose(0, 1)
        score = score.flatten(2)
        base_logit = torch.mean(score, dim=2)

        if self.T == 99:
            att_logit = torch.max(score, dim=2)[0]
        else:
            score_soft = self.softmax(score * self.T)
            att_logit = torch.sum(score * score_soft, dim=2)

        return base_logit + self.lam * att_logit


class MHA(nn.Module):
    temp_settings = {
        1: [1],
        2: [1, 99],
        4: [1, 2, 4, 99],
        6: [1, 2, 3, 4, 5, 99],
        8: [1, 2, 3, 4, 5, 6, 7, 99]
    }

    def __init__(self, num_heads, lam, input_dim, num_classes):
        super(MHA, self).__init__()
        self.temp_list = self.temp_settings[num_heads]
        self.multi_head = nn.ModuleList([
            DynamicCSRA(input_dim, num_classes, self.temp_list[i], lam)
            for i in range(num_heads)
        ])

    def forward(self, x):
        logit = 0.
        for head in self.multi_head:
            logit += head(x)
        return logit
