import torch
import numpy as np
import sys
import os
import cv2
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
from dataloader.face_detection import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default='/content/result_resnet18/model_59.bin', type=str)
    parser.add_argument("--model_name", default='resnet18', type=str)
    parser.add_argument("--svm_model_path",
                        default="/content/Facial-micro-expression-analysis-in-remote-chat/dataloader/finalized_face_detection_model.sav",
                        type=str)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--num_classes", default=7, type=int)
    parser.add_argument('--video_path', default='./Facial-micro-expression-analysis-in-remote-chat/data/sheldon.mp4', type=str)
    parser.add_argument('--video_name', default='sheldon.mp4', type=str)
    parser.add_argument('--sample_rate', default=10, type=int)
    parser.add_argument('--result_dir', default="./detection_results", type=str)

    args = parser.parse_args()
    
    # create result directory 
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    
    # image size
    img_size = 48
    
    # mapping {index : expression}
    class2lbl = {'0': 'angry', '1': 'disgust', '2': 'fear', '3': 'happy', '4': 'neutral', '5': 'sad', '6': 'surprise'}
    
    # sample frame sequences
    frames = sample_video(args.video_path, args.sample_rate)
    print("Sampled Frame Sequences: ", frames)
    
    # process frame
    # input :
    #  - frames : the sample frames
    #  - result_dir : output directory, "./detection_results"
    #  - image_size : (48, 48)
    #  - video_name : e.g. sheldon.mp4
    #  - model_path : /content/Facial-micro-expression-analysis-in-remote-chat/dataloader/finalized_face_detection_model.sav
    # output : 
    #  - resized : 
    resized = process_frame(frames, result_dir=args.result_dir, image_size=(img_size, img_size),
                            video_name=args.video_path.split("/")[-1], model_path=args.svm_model_path)
    print(resized.shape)

    if args.model_name == "resnet18":
        model = models.resnet18(pretrained=False, num_classes=args.num_classes)
    elif args.model_name == "resnet50":
        model = models.resnet50(pretrained=False, num_classes=args.num_classes)

    model.load_state_dict(torch.load(args.model_path))

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    model.eval()
    model.to(args.device)
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_imgs = torch.zeros((resized.shape[0], resized.shape[3], resized.shape[1], resized.shape[2]))
    for i in range(resized.shape[0]):
        print(resized[i])
        test_imgs[i] = test_transform(resized[i])
    print(test_imgs[0])
    test_imgs = test_imgs.to(args.device)

    out = model(test_imgs)
    # print(out)
    emotions = []
    emotions_lbl = []
    predictions = torch.argmax(out, axis=1)
    txt = open(os.path.join(args.result_dir, "emotions.txt"), "w")
    for i in range(predictions.shape[0]):
        p = predictions[i].cpu().detach().item()
        img_name = args.video_path.split("/")[-1].split(".")[0] + "_{}.png".format(i)
        cv2.imwrite(os.path.join(args.result_dir, img_name), resized[i])
        txt.write("img_name: {}, {}\n".format(img_name, class2lbl[str(p)]))
        emotions.append(class2lbl[str(p)])
        emotions_lbl.append(int(p))
    plot_trend(number_of_frames=resized.shape[0], emotion_labels=emotions_lbl, input_video_name=args.video_name, save_path=args.result_path)



if __name__ == "__main__":
    main()
