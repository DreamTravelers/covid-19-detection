# -*- coding: utf-8 -*-
import argparse
import os
import time

import torch
import torch.utils.data
from torch import optim

from ImageTextDataset import ImageDataset
from modal.vgg import vgg_random
from predication import train, test
from modal.textlstm import TextLstm, fuse_cv_text_model
from tool.save2File import SaveTool


def main():
    parser = argparse.ArgumentParser(description="covid-19-detection")
    parser.add_argument('--embed_dim', type=int, default=128, help='image size')
    parser.add_argument('--lr', type=int, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='max epoch')
    parser.add_argument('--weight_decay', type=int, default=5e-4, help='L2 weight decay')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')  # 8 16 32 464
    parser.add_argument('--test_batch_size', type=int, default=128, help='input batch size for testing')
    parser.add_argument('--use_cuda', type=bool, default=False, help='cuda')
    parser.add_argument('--save_result', type=bool, default=False, help='save result')
    parser.add_argument('--save_modal', type=bool, default=False, help='save result')

    args = parser.parse_args()

    modal = "vgg_random_TextLstm"

    now_time = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    info = str(args.embed_dim) + "_" + str(args.lr) + "_" + str(args.batch_size)
    file_name = modal + "_" + info + "_" + now_time + ".out"

    st = SaveTool(file_name)
    print("model", modal)
    print("embed_dim", str(args.embed_dim))
    print("lr", str(args.lr))
    print("batch_size", str(args.batch_size))
    print("epochs", str(args.epochs))
    if args.save_result:
        st.save2file("model : " + modal, "embed_dim : " + str(args.embed_dim), "lr : " + str(args.lr),
                     "batch_size : " + str(args.batch_size), "epochs : " + str(args.epochs))

    data_path = './data/'
    device = torch.device("cuda" if args.use_cuda else "cpu")

    if args.use_cuda:
        torch.backends.cudnn.benchmark = True
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    train_dataset = ImageDataset(data_path + "train.txt", "train", size=args.embed_dim)
    test_dataset = ImageDataset(data_path + "test.txt", "test", size=args.embed_dim)

    _train = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    _test = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True)

    image_net = vgg_random().to(device)
    text_net = TextLstm().to(device)
    image_text_net = fuse_cv_text_model(image_net, text_net).to(device)

    # Observe that all parameters are being optimized
    # optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(image_text_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    acc = 0

    for epoch in range(args.epochs + 1):
        train(image_text_net, _train, optimizer, epoch, acc, device, args.save_result, st)
        test_acc = test(image_text_net, _test, device, args.save_result, st)
        scheduler.step()

        if test_acc > acc:
            acc = test_acc

            if args.save_modal:
                if not os.path.isdir("./checkpoint"):
                    os.mkdir("./checkpoint")
                torch.save(image_text_net.state_dict(), './checkpoint/' + modal + '.pth')

    p_str = 'The best Accuracy ：%.5f' % acc
    print(p_str)
    if args.save_result:
        st.save2file(p_str)


if __name__ == '__main__':
    main()
