
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import glob
import pandas as pd
import torch.nn.functional as F


class CelebHQLoader(Dataset):

    def __init__(self, data_path : str, annotation_path : str, train_test_size : float,  is_train : bool ):
        self.data_path = data_path
        self.labels = pd.read_csv(annotation_path, sep=" ")[["img_name","No_Beard"]]
        self.clean_filtered_imgs()
        # self.make_balanced()

        if is_train:
            set_size = int(self.labels.shape[0]*train_test_size)  
            self.labels = self.labels.iloc[:set_size] 
            
        else:
            set_size = int(self.labels.shape[0]*(1-train_test_size))
            self.labels = self.labels.iloc[-set_size:]
        
        self.img_transform= T.Compose([
            T.Resize((224, 224)),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=5),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    

            
    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, id):
        img_name = self.labels["img_name"].iloc[id]
        img_path = self.data_path + "/" + img_name[:-4]+".png"
        img = self.img_transform(Image.open(img_path))

        label = torch.tensor(self.labels["No_Beard"].iloc[id])
        if label > 1:
            label = 1
        else:
            label = 0
        out = {"data":img, "label":label}
        return out

    def make_balanced(self):
        balanced_no_beard = self.labels[self.labels["No_Beard"]==0].sample(n=2000, random_state=1)
        self.labels = self.labels.drop(self.labels[self.labels.No_Beard == 0].index)
        self.labels = pd.concat([balanced_no_beard, self.labels])

    def clean_filtered_imgs(self):

        get_name = lambda x : x.split("/")[-1][:-4]+".jpg"

        imgs = list(glob.glob(f"{self.data_path}/*"))
        names = [get_name(n) for n in imgs]

        self.labels.set_index("img_name", inplace = True)
        self.labels = self.labels.loc[names]
        self.labels.reset_index(inplace=True)
     
