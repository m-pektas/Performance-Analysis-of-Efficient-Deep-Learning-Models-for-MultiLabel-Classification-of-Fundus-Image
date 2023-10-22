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
        self.fc3 = nn.Linear(84, 8)

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
        self.edd_head = nn.Sequential(nn.Linear(1000, 8))

    def forward(self, x):
        x = self.model(x)
        return self.edd_head(x)

class Resnet18Pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        # self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Sequential(
               nn.Linear(512, 8))

    def forward(self, x):
        return self.model(x)


class MobileNetV2Pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        self.edd_head = nn.Sequential(nn.Linear(1000, 8))

    def forward(self, x):
        x = self.model(x)
        return self.edd_head(x)
class MobileNetV3_S_Pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.mobilenet_v3_small(pretrained=True)
        self.edd_head = nn.Sequential(nn.Linear(1000, 8))

    def forward(self, x):
        x = self.model(x)
        return self.edd_head(x)

class MobileNetV3_L_Pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.mobilenet_v3_large(pretrained=True)
        self.edd_head = nn.Sequential(nn.Linear(1000, 8))

    def forward(self, x):
        x = self.model(x)
        return self.edd_head(x)

class SqueezeNet_10_Pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.squeezenet1_0(pretrained=True)
        self.edd_head = nn.Sequential(nn.Linear(1000, 8))

    def forward(self, x):
        x = self.model(x)
        return self.edd_head(x)

class SqueezeNet_11_Pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.squeezenet1_1(pretrained=True)
        self.edd_head = nn.Sequential(nn.Linear(1000, 8))

    def forward(self, x):
        x = self.model(x)
        return self.edd_head(x)

class Resnext50_32x4dPretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnext50_32x4d(pretrained=True)
        self.edd_head = nn.Sequential(nn.Linear(1000, 8))

    def forward(self, x):
        x = self.model(x)
        return self.edd_head(x)

class Resnext101_32x8dPretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnext101_32x8d(pretrained=True)
        self.edd_head = nn.Sequential(nn.Linear(1000, 8))

    def forward(self, x):
        x = self.model(x)
        return self.edd_head(x)   


class EfficientNetB3Pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.efficientnet_b3(pretrained=True)
        self.edd_head = nn.Sequential(nn.Linear(1000, 8))

    def forward(self, x):
        x = self.model(x)
        return self.edd_head(x)  

class EfficientNetB1Pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.efficientnet_b1(pretrained=True)
        self.edd_head = nn.Sequential(nn.Linear(1000, 8))

    def forward(self, x):
        x = self.model(x)
        return self.edd_head(x)  
    
class EfficientNetB2Pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.efficientnet_b2(pretrained=True)
        self.edd_head = nn.Sequential(nn.Linear(1000, 8))

    def forward(self, x):
        x = self.model(x)
        return self.edd_head(x)  

class EfficientNetB0Pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.efficientnet_b0(pretrained=True)
        self.edd_head = nn.Sequential(nn.Linear(1000, 8))

    def forward(self, x):
        x = self.model(x)
        return self.edd_head(x)  

class EfficientNetB4Pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.efficientnet_b4(pretrained=True)
        self.edd_head = nn.Sequential(nn.Linear(1000, 8))

    def forward(self, x):
        x = self.model(x)
        return self.edd_head(x)  
    
class EfficientNetV2MPretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.efficientnet_v2_l(pretrained=True)
        self.edd_head = nn.Sequential(nn.Linear(1000, 8))

    def forward(self, x):
        x = self.model(x)
        return self.edd_head(x)  

class ViTB16Pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.vit_b_16(pretrained=True)
        self.edd_head = nn.Sequential(nn.Linear(1000, 8))

    def forward(self, x):
        x = self.model(x)
        return self.edd_head(x)  
    

class SwinSPretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.swin_s(pretrained=True)
        self.edd_head = nn.Sequential(nn.Linear(1000, 8))

    def forward(self, x):
        x = self.model(x)
        return self.edd_head(x)  

class SwinTPretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.swin_t(pretrained=True)
        self.edd_head = nn.Sequential(nn.Linear(1000, 8))

    def forward(self, x):
        x = self.model(x)
        return self.edd_head(x)  

class Convnext_TinyPretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.convnext_tiny(pretrained=True)
        self.edd_head = nn.Sequential(nn.Linear(1000, 8))

    def forward(self, x):
        x = self.model(x)
        return self.edd_head(x)

def get_model(model_name, device, kwargs):
    model = None
    if model_name == "net":
        model =  Net().to(device)
    elif model_name == "Resnet50Pretrained":
        model =  Resnet50Pretrained().to(device)
    elif model_name == "MobileNetV2Pretrained":
        model = MobileNetV2Pretrained().to(device)
    elif model_name == "MobileNetV3_S_Pretrained":
        model = MobileNetV3_S_Pretrained().to(device)
    elif model_name == "MobileNetV3_L_Pretrained":
        model = MobileNetV3_L_Pretrained().to(device)
    elif model_name == "SqueezeNet_10_Pretrained":
        model = SqueezeNet_10_Pretrained().to(device)
    elif model_name == "SqueezeNet_11_Pretrained":
        model = SqueezeNet_11_Pretrained().to(device)
    elif model_name == "Resnet18Pretrained":
        model = Resnet18Pretrained().to(device)
    elif model_name == "Resnext50_32x4dPretrained":
        model = Resnext50_32x4dPretrained().to(device)
    elif model_name == "Resnext101_32x8dPretrained":
        model = Resnext101_32x8dPretrained().to(device)
    elif model_name == "Convnext_TinyPretrained":
        model = Convnext_TinyPretrained().to(device)
    elif model_name == "EfficientNetB3Pretrained":
        model = EfficientNetB3Pretrained().to(device)
    elif model_name == "EfficientNetB1Pretrained":
        model = EfficientNetB1Pretrained().to(device)
    elif model_name == "EfficientNetB2Pretrained":
        model = EfficientNetB2Pretrained().to(device)
    elif model_name == "EfficientNetB0Pretrained":
        model = EfficientNetB0Pretrained().to(device)
    elif model_name == "EfficientNetB4Pretrained":
        model = EfficientNetB4Pretrained().to(device)
    elif model_name == "EfficientNetV2MPretrained":
        model = EfficientNetV2MPretrained().to(device)
    elif model_name == "ViTB16Pretrained":
        model = ViTB16Pretrained().to(device)
    elif model_name == "SwinSPretrained":
        model = SwinSPretrained().to(device)
    elif model_name == "SwinTPretrained":
        model = SwinTPretrained().to(device)
    else:
        raise Exception("Model not found")


    if kwargs.get("load_model"):
        try:
            model.load_state_dict(torch.load(kwargs.get("load_model_path")))
            print("Model loaded from path: {}".format(kwargs.get("load_model_path")))
        except:
            raise Exception("Model not found in path: {}".format(kwargs.get("load_model_path")))
    return model
