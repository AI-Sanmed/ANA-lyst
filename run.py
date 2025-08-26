#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
from torchvision import transforms
from models.res2net_v1b_csra import Res2NetV1bCsraMLC, Res2NetV1bCsraMTML

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def read_image(image_file):
    image = Image.open(image_file).convert('RGB')
    return image


def predict_pn(image_ori):
    transform = transforms.Compose([
        transforms.Resize(896, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(**{
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'inplace': False,
        }),
    ])
    image = transform(image_ori).unsqueeze(0)

    weights = 'weights/ANA_PN_Res2NetV1bCsra_v2.0.0.250411.pth.tar'
    model = Res2NetV1bCsraMLC(1, 0.1, 1, input_dim=2048)
    model.load_state_dict(torch.load(weights, map_location='cuda')['state_dict'])
    print(f'| Load Weights: {os.path.basename(weights)}')
    model.to('cuda')
    model.eval()

    with torch.no_grad():
        image = image.to(torch.device('cuda'))
        image = Variable(image)
        logits = model(image)

        probabilities = torch.sigmoid(logits)
        predicted = torch.where(probabilities >= 0.177423, 1, 0)
        probabilities = probabilities.cpu().detach().numpy()[0][0]
        predicted = predicted.cpu().detach().numpy()[0][0]

    return predicted, probabilities


def predict_complex_pattern(image_ori):
    titer_list = [80, 160, 320, 640, 1280]

    transform = transforms.Compose([
        transforms.Resize(896, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(**{
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'inplace': False,
        }),
    ])
    image = transform(image_ori).unsqueeze(0)

    weights = 'weights/ANA_MTMlcReg_Res2NetV1bCsra_v1.4.1.250503.pth.tar'
    model = Res2NetV1bCsraMTML(1, 0.1, 14, input_dim=2048)
    model.load_state_dict(torch.load(weights, map_location='cuda')['state_dict'])
    print(f'| Load Weights: {os.path.basename(weights)}')
    model.to('cuda')
    model.eval()

    with torch.no_grad():
        image = image.to(torch.device('cuda'))
        image = Variable(image)
        logits, regressions = model(image)

        probabilities = torch.sigmoid(logits)
        predicted = torch.where(probabilities >= 0.5, 1, 0)
        probabilities = probabilities.cpu().detach().numpy()[0]
        predicted = predicted.cpu().detach().numpy()[0]
        regressions = regressions.cpu().detach().numpy()[0]

    result = []
    for idx, p in enumerate(predicted):
        if p:
            probability = probabilities[idx]
            reg = regressions[idx]
            regression = round(reg * 102.13236949494936 + 20.468839336763864, 4)
            titer = titer_list[np.argmin(np.abs(regression - np.array(titer_list)))]
            titer = str(titer).zfill(4)
            result.append([idx, probability, titer, regression])
    result = np.array(result)

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ANA-lyst')
    parser.add_argument('--image_file', default='images/1794.png')
    args = parser.parse_args()

    image = read_image(args.image_file)

    pn_predicted, pn_probabilities = predict_pn(image)
    print(pn_predicted, pn_probabilities)

    if pn_predicted == 1:
        complex_pattern = predict_complex_pattern(image)
        print(complex_pattern)
