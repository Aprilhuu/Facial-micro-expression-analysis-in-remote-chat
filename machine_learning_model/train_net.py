import torch
import numpy as np
import sys
import os

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


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class Trainer:
    def __init__(self, hparam):
        self.start_epoch = hparam['start_epoch']
        self.eigen = hparam['eigen']
        self.device = hparam['device']
        self.lr = hparam['lr']
        self.model_name = hparam['model_name']
        self.bs = hparam['batchsize']
        self.logspace = hparam['logspace']
        self.gamma = hparam['gamma']
        self.momentum = hparam['momentum']
        self.weight_decay = hparam['weight_decay']
        self.epoch = hparam['epoch']
        self.seed = hparam['seed']
        self.opt = hparam['opt']
        self.datapath = hparam['datapath']
        self.result_path = hparam['result_path']
        self.lst = [[], [], [], [], [], [], [], []]
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.device)
        self.classes = os.listdir(os.path.join(hparam['datapath'], "train"))
        self.classes.sort()
        self.num_classes = len(os.listdir(os.path.join(hparam['datapath'], "train")))
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        seed_torch(int(self.seed))

        self.prepare_model()
        self.prepare_dataset()

    def prepare_model(self):
        if self.model_name == "resnet18":
            model = models.resnet18(pretrained=False, num_classes=self.num_classes)
            self.img_size = 224
        elif self.model_name == "resnet50":
            model = models.resnet50(pretrained=False, num_classes=self.num_classes)
            self.img_size = 224
        self.model = model
        if self.opt == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum,
                                             weight_decay=self.weight_decay)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if self.start_epoch != 0:
            self.model.load_state_dict(
                torch.load(os.path.join(self.result_path, "model_{}.bin".format(self.start_epoch - 1))))
            self.lst = torch.load(os.path.join(self.result_path, "list_{}.bin".format(self.start_epoch - 1)))

        self.loss_func = nn.CrossEntropyLoss(reduction="mean")

    def prepare_dataset(self):
        ## Data augmentation
        if self.eigen:
            train_transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            test_transform = transforms.Compose([
                transforms.Resize((48, 48)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomCrop(self.img_size),
                transforms.RandomRotation(degrees=(-90, 90)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            test_transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                # transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        self.train_set = datasets.ImageFolder(root=self.datapath + "/train", transform=train_transform)
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.bs, shuffle=True)
        self.test_set = datasets.ImageFolder(root=self.datapath + "/test", transform=test_transform)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=self.bs, shuffle=True, )

    def train(self):
        self.model.train()
        self.model.to(self.device)
        torch.autograd.set_detect_anomaly(True)
        Loss = 0
        Acc = 0
        for i, (img, lbl) in enumerate(self.train_loader):
            img = img.to(self.device)
            lbl = lbl.to(self.device)

            output = self.model(img)
            lam = 0.5
            l2_reg = 0
            for param in self.model.parameters():
                l2_reg += torch.norm(param)
            loss = self.loss_func(output, lbl)
            reg_loss = loss + lam * l2_reg
            self.optimizer.zero_grad()
            reg_loss.backward()
            self.optimizer.step()
            # print(output.size())
            prediction = torch.argmax(output, axis=1)

            accuracy = torch.sum(prediction == lbl).item() / len(prediction)
            Acc += accuracy
            Loss += loss.cpu().item()

            if i % 10 == 0:
                print("[train] batch: %d, loss: %.3f, acc: %.3f" % (i + 1, Loss / (i + 1), Acc / (i + 1)))
        self.lst[0].append(Loss / (i + 1))
        self.lst[1].append(Acc / (i + 1))

    def draw(self):
        x = np.arange(0, len(self.lst[0]), 1)
        train_l = np.array(self.lst[0])
        train_e = np.array(self.lst[1])
        test_l = np.array(self.lst[2])
        test_e = np.array(self.lst[3])
        plt.figure()
        plt.subplot(211)
        plt.plot(x, train_l, color='red')
        plt.plot(x, test_l, color='blue')
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.subplot(212)
        plt.plot(x, train_e, color='red')
        plt.plot(x, test_e, color='blue')
        plt.xlabel("epoch")
        plt.ylabel("acc")

        plt.savefig(os.path.join(self.result_path, "curve.png"))
        plt.close()

    def test(self):
        self.model.eval()
        self.model.to(self.device)

        Loss = 0
        Acc = 0
        for i, (img, lbl) in enumerate(self.test_loader):
            img = img.to(self.device)
            lbl = lbl.to(self.device)

            output = self.model(img)
            loss = self.loss_func(output, lbl)
            # print(output.size())
            prediction = torch.argmax(output, axis=1)

            accuracy = torch.sum(prediction == lbl).item() / len(prediction)
            Acc += accuracy
            Loss += loss.cpu().item()

            if i % 10 == 0:
                print("[test] batch: %d, loss: %.3f, acc: %.3f" % (i + 1, Loss / (i + 1), Acc / (i + 1)))
        self.lst[2].append(Loss / (i + 1))
        self.lst[3].append(Acc / (i + 1))

    def test_eigen(self):
        self.model.eval()
        self.model.to(self.device)

        Loss = 0
        Acc = 0
        for i, (img, lbl) in enumerate(self.test_loader):
            # img = img.to(self.device)
            lbl = lbl.to(self.device)
            all_output = torch.zeros((self.num_classes, img.shape[0], self.num_classes))
            for cls_idx in range(self.num_classes):
                cls = self.classes[cls_idx]
                filtered_batch = torch.zeros((img.shape[0], 3, 48, 48)).to(self.device)
                for img_idx in range(img.shape[0]):
                    individual_img = img[img_idx][0] * 255
                    eigen_img = (eigenface_filter(individual_img.cpu(), os.path.join(os.getcwd(), 'pickle') + "/", cls,
                                                  proj_basis_num=160))
                    eigen_img_rgb = transforms.ToTensor()(np.stack((eigen_img,) * 3, axis=-1))
                    # print(eigen_img_rgb.shape)
                    filtered_batch[img_idx] = eigen_img_rgb
                filtered_batch = F.interpolate(filtered_batch, size=(self.img_size, self.img_size), mode='bilinear')
                output = self.model(filtered_batch)
                # print(output.shape)
                all_output[cls_idx] = output
            best_output = torch.max(all_output, axis=0)[0].to(self.device)
            loss = self.loss_func(best_output, lbl)
            prediction = torch.argmax(best_output, axis=1)

            accuracy = torch.sum(prediction == lbl).item() / len(prediction)
            Acc += accuracy
            Loss += loss.cpu().item()

            if i % 10 == 0:
                print("[test] batch: %d, loss: %.3f, acc: %.3f" % (i + 1, Loss / (i + 1), Acc / (i + 1)))
        self.lst[2].append(Loss / (i + 1))
        self.lst[3].append(Acc / (i + 1))

    def start(self):
        if self.logspace != 0:
            logspace_lr = np.logspace(np.log10(self.lr), np.log10(self.lr) - self.logspace, self.epoch)
        print(self.start_epoch, self.epoch)
        for e in range(self.start_epoch, self.epoch):
            print(e)
            if self.logspace != 0:
                for param in self.optimizer.param_groups:
                    param['lr'] = logspace_lr[e]
            self.train()
            if self.eigen:
                self.test_eigen()
            else:
                self.test()
            self.draw()

        self.draw()
        torch.save(self.model.state_dict(), os.path.join(self.result_path, "model_{}.bin".format(self.epoch - 1)))
        torch.save(self.lst, os.path.join(self.result_path, "list_{}.bin".format(self.epoch - 1)))


def main():
    print("hello")
    stream = open('./Facial-micro-expression-analysis-in-remote-chat/machine_learning_model/hparam.yaml', 'r')
    hparam = yaml.load(stream)
    print(hparam)

    trainer = Trainer(hparam)
    trainer.start()


if __name__ == "__main__":
    main()