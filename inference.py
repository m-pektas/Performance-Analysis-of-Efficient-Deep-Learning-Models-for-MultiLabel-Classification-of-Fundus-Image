import torchvision.transforms as T
from PIL import Image
import argparse
from network.model import get_model 
import torch


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Trainer')
    
    parser.add_argument('--model_name', default="EfficientNetB3Pretrained")
    parser.add_argument('--model_path', default="logs/exp_2/tb_2022_11_11-03:19:29_PM/models/net_best_epoch_1__iter_80__loss_1.3364__acc_0.45.pth")
    parser.add_argument('--device', default="cuda")
    parser.add_argument('--image_path', default="data/preprocessed_images/0_left.jpg")
    args = parser.parse_args()
    params = vars(args)
    print(params)


    model = get_model(args.model_name, args.device, {})
    model.load_state_dict(torch.load(args.model_path))
    model.eval()


    img_transform= T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    classes = {
        0 : "N", 
        1 : "D",
        2 : "G",
        3 : "C", 
        4 : "A",
        5 : "H",
        6 : "M", 
        7 : "O"
        }


    
    img = Image.open(args.image_path)
    img = img_transform(img).unsqueeze(0).to(args.device)
    output = model(img)
    pred = output.argmax(dim=1, keepdim=True).item()

    print("Prediction : ",classes[pred])