import torch
from torch.xpu import device
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

# torch.set_default_device('cuda:0')


class ImgClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # [batch,1,28,28] -> [batch,100,28,28] 如果没有padding就是26，26
        self.cnn=nn.Conv2d(
            in_channels=1,  # 输入通道
            out_channels=100,  # channels*conv kernel
            kernel_size=(3,3),  # 卷积核大小
            stride=1,  # 步长
            padding=1,  # 可以用数值，也可以用字符串same or valid，一般用卷积核大小整除2
        )
        self.act=nn.ReLU()
        self.flatten=nn.Flatten()
        self.output=nn.Linear(100*28*28,10)

    def forward(self, input_data):
        # input_data  [batch_size,features] -> [batch_size,256]
        out=self.cnn(input_data)
        out=self.act(out)
        out=self.flatten(out)
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
    transform=ToTensor(),
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
test_dl = DataLoader(test_data, batch_size=2,)
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
    pbar=tqdm(train_dl)
    for img, lbl in pbar:
        logits = model(img)
        loss = loss_fn(logits, lbl)
        loss.backward()
        optimizer.step()
        model.zero_grad()
        pbar.set_description(f"epoch{epoch+1},loss{loss.item():.4f}")
    correct = 0
    total_loss = 0
    with torch.no_grad():  # 测试不需要训练，就no_grad()

        for img, lbl in test_dl:
            logits = model(img)
            loss = loss_fn(logits, lbl)
            total_loss += loss.item()
            a=logits.argmax() ==lbl
            correct += sum(a)  # 在最后一个维度求概率
        print('accuracy', correct / len(test_data))  # 准确率
        print('loss', total_loss / len(test_dl))
