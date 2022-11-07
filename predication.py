# -*- coding: utf-8 -*-
"""
@author: liulin
@time: 2021/4/6 15:55
@subscribe:

"""
from datetime import datetime
from torch import nn


def train(model, _train, optimizer, epoch, acc, device, save_result, st):
    model.train()
    loss_list, batch_list = [], []
    criterion = nn.CrossEntropyLoss()

    for i, data in enumerate(_train, 0):
        train_img, train_label, desc = data

        optimizer.zero_grad()

        output = model(train_img, desc.squeeze(1))

        loss = criterion(output.to(device), train_label.long().to(device))

        loss_list.append(loss.detach().cpu().item())
        batch_list.append(i + 1)

        if i % 10 == 0:
            p_str = '%s Training:  Epoch %d, Batch: %d, Loss: %f , The best test Accuracy ：%f' % (
                datetime.now(), epoch, i, loss.detach().cpu().item(), acc)
            print(p_str)
            if save_result:
                st.save2file(p_str)

        loss.backward()
        optimizer.step()
    return 0


def test(model, test_loader, device, save_result, st):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_correct = 0
    count = 0
    avg_loss = 0.0
    for i, (images, labels, desc) in enumerate(test_loader):
        output = model(images.to(device), desc.squeeze())
        avg_loss += criterion(output.detach().cpu(), labels.long().detach().cpu()).sum()
        pred = output.detach().argmax(1)
        total_correct += pred.eq(labels.to(device).view_as(pred.to(device))).sum()
        count += len(labels)

    avg_loss /= len(test_loader)
    p_str = 'Test Avg. Loss: %f, correct / total : %f / %f , Accuracy: %f' % (
        avg_loss.detach().cpu().item(), float(total_correct), count,
        float(total_correct) / count)
    print(p_str)
    if save_result:
        st.save2file(p_str)

    return float(total_correct) / count
