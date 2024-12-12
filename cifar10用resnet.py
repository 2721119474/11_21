#迁移学习
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet通常需要固定大小的输入
    transforms.ToTensor(),#将图像转换为pytorch张量
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载CIFAR-10数据集
train_dataset = datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform)
#train=True 表示如果数据不存在则从互联网下载
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 加载预训练的ResNet模型
model = models.resnet18(pretrained=True)

# 修改最后一层以适应CIFAR-10的分类任务
num_ftrs = model.fc.in_features #最后一层全连接层输入特征数
model.fc = nn.Linear(num_ftrs, 10) ## 将原来的全连接层替换为新的线性层，输入特征数保持不变，输出类别数改为10


# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 训练模型
def train(model, train_loader, criterion, optimizer, device):
    model.train()  # 将模型设置为训练模式
    running_loss = 0.0  # 初始化累计损失
    correct = 0  # 初始化正确预测的数量
    total = 0  # 初始化总样本数量

    for images, labels in tqdm(train_loader):  # 遍历训练数据加载器
        images, labels = images.to(device), labels.to(device)  # 将图像和标签移动到指定设备（如GPU）

        optimizer.zero_grad()  # 清零梯度
        outputs = model(images)  # 通过模型前向传播获取输出
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数

        running_loss += loss.item() * images.size(0)  # 累加当前批次的损失
        _, predicted = outputs.max(1)  # 获取每个样本的最大值索引作为预测结果
        total += labels.size(0)  # 累加当前批次的样本数量
        correct += predicted.eq(labels).sum().item()  # 累加正确预测的数量

    epoch_loss = running_loss / len(train_loader.dataset)  # 计算平均损失
    epoch_acc = correct / total  # 计算准确率
    return epoch_loss, epoch_acc  # 返回平均损失和准确率



# 测试模型
def test(model, test_loader, criterion, device):
    model.eval()  # 将模型设置为评估模式
    running_loss = 0.0  # 初始化累计损失
    correct = 0  # 初始化正确预测的数量
    total = 0  # 初始化总样本数量

    with torch.no_grad():  # 禁用梯度计算以节省内存和计算资源
        for images, labels in tqdm(test_loader):  # 遍历测试数据加载器
            images, labels = images.to(device), labels.to(device)  # 将图像和标签移动到指定设备（如GPU）
            outputs = model(images)  # 通过模型前向传播获取输出
            loss = criterion(outputs, labels)  # 计算损失

            running_loss += loss.item() * images.size(0)  # 累加当前批次的损失
            _, predicted = outputs.max(1)  # 获取每个样本的最大值索引作为预测结果
            total += labels.size(0)  # 累加当前批次的样本数量
            correct += predicted.eq(labels).sum().item()  # 累加正确预测的数量

    epoch_loss = running_loss / len(test_loader.dataset)  # 计算平均损失
    epoch_acc = correct / total  # 计算准确率
    return epoch_loss, epoch_acc  # 返回平均损失和准确率


model.to(device)
# 训练和评估循环
num_epochs = 10
for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = test(model, test_loader, criterion, device)
    print(
        f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
