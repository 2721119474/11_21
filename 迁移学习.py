from torchvision.models import efficientnet_b0
import torch.nn as nn
import torch
model = efficientnet_b0(weights='DEFAULT')


custom_classifier=nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(1280,10)
)
model.features.trainable=False  # 冻结层，不训练
model.classifier=custom_classifier
input_data=torch.rand(size=(10,3,25,25))
result=model(input_data)
print(result.shape)
print(model)
