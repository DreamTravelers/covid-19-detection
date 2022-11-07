import copy

import torch
from PIL import Image
from torchvision import transforms

from ImageDataset import Resize, RandomRotate
from Net import Net
import torchvision.models as models

# mucca 0; pecora 1; gallina 2; scoiattolo 3; cane 4; ragno 5;  farfalla 6; cavallo 7; elefante 8; gatto 9;
# from dla import DLA
from dla import DLA
from resnet import ResNet50, ResNet18, ResNet34
import os
from lenet import LeNet5

# result = ["牛", "羊", "鸡", "松鼠", "狗", "蜘蛛", "蝴蝶", "马", "大象", "猫","牛", "羊", "鸡", "松鼠", "狗", "蜘蛛", "蝴蝶", "马", "大象", "猫"]
home = os.path.expanduser('~')
BASE = home + '/pycharmProject/imageClass/data/'
traindata_path = BASE + 'Flickr'
result = os.listdir(traindata_path)

img_path = 'data/train/45299.png'  # OIP-Tpo_INQU0iJ5Jl9n7JptXwHaFj.jpeg
img = Image.open(img_path).convert('RGB')
img = img.resize((32, 32), Image.BILINEAR)
size = 32

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
])

img = train_transform(img).reshape(1, 3, size, size)

# net = torch.load("./checkpoint/Flickr_DLA.pth", map_location='cpu')
# net = net.modules()
net = DLA()
net.eval()
path_checkpoint = "./checkpoint/Flickr_DLA.pth"  # 断点路径
checkpoint = torch.load(path_checkpoint, map_location='cpu')  # 加载断点
net.load_state_dict(checkpoint)  # 加载模型可学习参数

output = net(img)
res = torch.softmax(output, dim=1)
res = res.reshape(-1).detach().numpy().tolist()
res_alter = copy.deepcopy(res)

print("预测结果\n")
max_res = max(res)
max_index = res.index(max_res)
print(result[max_index], ": ", int(max_res * 10000) / 100.0, "%")

res.remove(max_res)
next_res = max(res)
next_index = res_alter.index(next_res)
print(result[next_index], ": ", int(next_res * 10000) / 100.0, "%")

res.remove(next_res)
last_res = max(res)
last_index = res_alter.index(last_res)
print(result[last_index], ": ", int(last_res * 10000) / 100.0, "%")
