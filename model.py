import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# from torchvision.models import resnet50, ResNet50_Weights, mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision import models
class Resnet50Pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        # self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model = models.resnext50_32x4d(pretrained=True)
        self.model.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 6))

    def forward(self, x):
        return self.model(x)

class Resnet18Pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        # self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Sequential(
               nn.Linear(512, 6))

    def forward(self, x):
        return self.model(x)


class MobileNetV2Pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        self.fc = nn.Sequential(
               nn.Linear(1000, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 2))

    def forward(self, x):
        x = self.model(x)
        return self.fc(x)
           
def get_model(model_name, device, kwargs):
    model = None
    if model_name == "net":
        model =  Net().to(device)
    elif model_name == "Resnet50Pretrained":
        model =  Resnet50Pretrained().to(device)
    elif model_name == "MobileNetV2Pretrained":
        model = MobileNetV2Pretrained().to(device)
    elif model_name == "Resnet18Pretrained":
        model = Resnet18Pretrained().to(device)
    else:
        raise Exception("Model not found")


    if kwargs.get("load_model"):
        try:
            model.load_state_dict(torch.load(kwargs.get("load_model_path")))
            print("Model loaded from path: {}".format(kwargs.get("load_model_path")))
        except:
            raise Exception("Model not found in path: {}".format(kwargs.get("load_model_path")))
    return model
