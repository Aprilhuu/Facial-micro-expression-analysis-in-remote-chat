import torch
import numpy as np
import sys
import os
from PIL import Image

proj_dir = os.path.join(os.getcwd(), "Facial-micro-expression-analysis-in-remote-chat")
sys.path.append(proj_dir)
from eigenface.eigenface import *
from torch import nn
from torchvision import models
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
import random
from torch import nn
import matplotlib.pyplot as plt
from google.colab import files
import yaml
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default='/content/result_resnet18/model_59.bin', type=str)
    parser.add_argument("--test", default=False, type=bool)
    parser.add_argument("--test_img_dir", default='/content/detection_results', type=str)
    parser.add_argument("--model_name", default='resnet18', type=str)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--data_path", default="./FER2013/test", type=str)

    args = parser.parse_args()

    class2lbl = {}
    classes = os.listdir(args.data_path)
    classes.sort()
    for i in range(len(classes)):
        class2lbl[str(i)] = classes[i]
    num_classes = len(classes)

    if args.model_name == "resnet18":
        model = models.resnet18(pretrained=False, num_classes=num_classes)
    elif args.model_name == "resnet50":
        model = models.resnet50(pretrained=False, num_classes=num_classes)

    model.load_state_dict(torch.load(args.model_path))

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

    model.to(args.device)
    img_size = 48
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_set = datasets.ImageFolder(root=args.data_path, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

    Acc = 0
    class_correct = {}
    class_total = {}
    for key in class2lbl:
        class_correct[key] = 0
        class_total[key] = len(os.listdir(os.path.join(args.data_path, class2lbl[key])))
    if args.test:
        for i, (img, lbl) in enumerate(test_loader):
            img = img.to(args.device)
            lbl = lbl.to(args.device)

            output = model(img)
            # print(output.size())
            prediction = torch.argmax(output, axis=1)
            for j in range(prediction.shape[0]):
                p = prediction[j]
                if p == lbl[j]:
                    class_correct[str(p.cpu().item())] += 1
            accuracy = torch.sum(prediction == lbl).item() / len(prediction)
            Acc += accuracy

        print("overall acc: {}".format(Acc / (i + 1)))
        for k in class_correct:
            print("{}: {} / {}".format(class2lbl[k], class_correct[k], class_total[k]))
    else:
        test_imgs = os.listdir(args.test_img_dir)
        test_imgs.sort()
        for i in range(len(test_imgs)):
            img_path = os.path.join(args.test_img_dir, test_imgs[i])
            img_arr = Image.open(img_path).convert('RGB')
            img_arr = test_transform(img_arr)[None, :, :, :].to(args.device)
            p = torch.argmax(model(img_arr)).cpu().item()

            print(test_imgs[i], class2lbl[str(p)])


if __name__ == "__main__":
    main()

