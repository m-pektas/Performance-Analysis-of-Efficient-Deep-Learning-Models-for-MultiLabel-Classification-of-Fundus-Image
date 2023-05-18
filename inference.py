import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models
import torchvision.transforms as T
from PIL import Image

class MobileNetV2Pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.mobilenet_v2()
        self.fc = nn.Sequential(
               nn.Linear(1000, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 2))

    def forward(self, x):
        x = self.model(x)
        return self.fc(x)




if __name__ == "__main__":


    model = MobileNetV2Pretrained()
    model.load_state_dict(torch.load("logs/exp_5binaryfixed/tb_2022_11_26-01:42:39_AM/models/net_best_epoch_44__iter_170__loss_0.3678__acc_0.95.pth"))
    model.eval()


    img_transform= T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    classes = {0 : "Without Facial Hair", 1 : "With Facial Hair"}
    
    img = Image.open("data/Celaba_StyleganAligned/012065.png")
    img = img_transform(img).unsqueeze(0)
    output = model(img)
    pred = output.argmax(dim=1, keepdim=True).item()

    print("Prediction : ",classes[pred])