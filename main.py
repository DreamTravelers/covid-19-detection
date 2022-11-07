# -*- coding: utf-8 -*-
import argparse
import os

import torch
import torch.utils.data
from torch import nn, optim

from ImageDataset import ImageDataset
from predication import train, test
from modal.resnet import ResNet18
from modal.vgg import vgg16


def main():
    parser = argparse.ArgumentParser(description="imageClass")
    parser.add_argument('--embed_dim', type=int, default=128, help='image size')  # 32 128 256 512
    parser.add_argument('--lr', type=int, default=0.1, help='learning rate')  # 0.1 0.01 0.001 0.0001
    parser.add_argument('--epochs', type=int, default=100, help='max epoch')
    parser.add_argument('--weight_decay', type=int, default=0.0001, help='L2 weight decay')  # 0.0001 0.0002 0.0005 0.001
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')  # 8 16 32 464
    parser.add_argument('--test_batch_size', type=int, default=8, help='input batch size for testing')
    parser.add_argument('--use_cuda', type=bool, default=True, help='cuda')

    args = parser.parse_args()

    print("model", "vgg16")
    print("embed_dim", str(args.embed_dim))
    print("lr", str(args.lr))
    print("batch_size", str(args.batch_size))
    print("epochs", str(args.epochs))

    data_path = './data/'
    device = torch.device("cuda" if args.use_cuda else "cpu")

    if args.use_cuda:
        torch.backends.cudnn.benchmark = True
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    train_dataset = ImageDataset(data_path + "train.txt", "train", size=args.embed_dim)
    test_dataset = ImageDataset(data_path + "test.txt", "test", size=args.embed_dim)

    _train = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    _test = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=2)

    net = ResNet18().to(device)

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    acc = 0

    for epoch in range(args.epochs + 1):
        train(net, _train, optimizer, epoch, acc, device)
        test_acc = test(net, _test, device)
        scheduler.step()

        if test_acc > acc:
            acc = test_acc

            if not os.path.isdir("./checkpoint"):
                os.mkdir("./checkpoint")
            torch.save(net.state_dict(), './checkpoint/vgg16.pth')

    print('The best Accuracy ：%.5f' % acc)


if __name__ == '__main__':
    main()
