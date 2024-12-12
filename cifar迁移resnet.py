# 使用预训练模型权重，使用Transfor learn训练cifar识别模型、验证（.py+训练截图）
from torchvision.models import resnet18
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import torch

model = resnet18(weights='DEFAULT')


custom_classifier=nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(512,128),
    nn.ReLU(),
    nn.Linear(128,10)
)

model.conv1.tranable=False
model.bn1.tranable=False
model.relu.tranable=False
model.maxpool.tranable=False
model.layer1.tranable=False
model.layer2.tranable=False
model.layer3.tranable=False
model.layer4.tranable=False
model.avgpool.tranable=False
model.fc=custom_classifier

train_data = CIFAR10(
    root="../data/cifar_10",
    train=True,
    download=True,
    transform=ToTensor()
)
test_data = CIFAR10(
    root="../data/cifar_10",
    train=False,
    download=True,
    transform=ToTensor()
)

model.to('cuda')
train_dl=DataLoader(train_data,batch_size=32,shuffle=True)
test_dl=DataLoader(test_data,batch_size=32,shuffle=False)

# 创建优化器
optimizer=Adam(model.parameters())
loss_fn=nn.CrossEntropyLoss()

for epoch in range(10):
    pbar=tqdm(train_dl)
    for img,lbl in pbar:
        img,lbl=img.to('cuda'),lbl.to('cuda')
        logits=model(img)
        loss=loss_fn(logits,lbl)
        loss.backward()
        optimizer.step()
        model.zero_grad()
        pbar.set_description(f"epoch{epoch+1},loss{loss.item():.4f}")
    correct = 0
    total_loss = 0
    total = 0
    with torch.no_grad():
        for img, lbl in test_dl:
            img, lbl = img.to('cuda'), lbl.to('cuda')
            logits = model(img)
            loss = loss_fn(logits, lbl)
            total_loss += loss.item()
            correct += (logits.argmax(dim=-1) == lbl).sum().item()
            total += lbl.size(0)
        print('acc', correct / total)
        print('loss', total_loss / len(test_dl))

