import torch
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam


class ImgClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(in_features=784, out_features=256)
        self.act = nn.ReLU()
        self.output = nn.Linear(256, 10)
        self.flatten = nn.Flatten()

    def forward(self, input_data):
        # input_data  [batch_size,features] -> [batch_size,256]
        out = self.flatten(input_data)
        out = self.hidden(out)
        out = self.act(out)
        return self.output(out)


train_data = FashionMNIST(
    root="../data/fashon_mnist",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = FashionMNIST(
    root="../data/fashon_mnist",
    train=False,
    download=True,
    transform=ToTensor()
)


# fun=lambda x:x.reshape(-1,784)
# print(type(fun))  # 函数也是python的数据类型，可执行

def dataproc(batch_data):
    # collate_data=[]  # 返回的数据类型不是想要的
    imgs, lbls = [], []
    for img, lbl in batch_data:
        img.reshape(-1, 784)
        # collate_data.append((img,lbl))
        imgs.append(img)
        lbls.append(lbl)
    return imgs, torch.tensor(lbls)


# train_dl=DataLoader(train_data,batch_size=2,shuffle=True,collate_fn=dataproc)  # 回调自定义函数
train_dl = DataLoader(train_data, batch_size=2, shuffle=True)
test_dl = DataLoader(test_data, batch_size=2)
# for img,lbl in train_dl:
#     print(img.shape)
model = ImgClassifier()

# 优化器
optimizer = Adam(model.parameters())  # 传入模型的参数
# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 训练 forward->backward
# 评估 validation
for epoch in range(3):
    for img, lbl in train_dl:
        logits = model(img)
        loss = loss_fn(logits, lbl)
        loss.backward()
        optimizer.step()
        model.zero_grad()
    correct = 0
    total_loss = 0
    with torch.no_grad():  # 测试不需要训练，就no_grad()

        for img, lbl in test_dl:
            logits = model(img)
            loss = loss_fn(logits, lbl)
            total_loss += loss.item()
            correct += sum(logits.argmax() == lbl)  # 在最后一个维度求概率
        print('accuracy', correct / len(test_data))  # 准确率
        print('loss', total_loss / len(test_dl))
