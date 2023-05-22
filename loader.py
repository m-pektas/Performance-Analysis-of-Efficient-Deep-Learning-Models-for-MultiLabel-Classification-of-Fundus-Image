
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import glob
import pandas as pd
import torch.nn.functional as F
import numpy as np
from loguru import logger as printer
class ODIR5K(Dataset):

    def __init__(self, data_path : str, annotation_path : str, train_test_size : float,  is_train : bool, augment : bool = False ):
        self.data_path = data_path
        self.annotation_path = annotation_path

        def set_filepath(x):
            return data_path + "/" + x
    
        def set_label(x):
            arr = ','.join(e for e in x if e.isalnum()).split(",")
            arr = np.asarray(arr).astype(int)
            result = np.where(arr == 1)
            return result[0][0]

        df = pd.read_csv(annotation_path)[["target","filename"]]
        df = df.sample(frac = 1)
        df.filename = df.filename.apply(set_filepath)
        df.target = df.target.apply(set_label)

       
        # self.make_balanced()

        if is_train:
            set_size = int(df.target.shape[0]*train_test_size)  
            df = df.head(set_size)
            printer.info(f"Train set size: {set_size}")
            printer.info(f"Train set distribution: {df.target.value_counts()}")
            
        else:
            set_size = int(df.target.shape[0]*(1-train_test_size))
            df = df.tail(set_size)
            printer.info(f"Test set size: {set_size}")
            printer.info(f"Test set distribution: {df.target.value_counts()}")
        
        self.df = df
        

        if augment:
            self.img_transform= T.Compose([
                T.Resize((224, 224)),
                 T.RandomEqualize(0.5),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

        else:
            self.img_transform= T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
    
    
            
    def __len__(self):
        return self.df.target.shape[0]

    def __getitem__(self, id):
        target, filename = self.df.iloc[id]
        img = self.img_transform(Image.open(filename))
        label = torch.tensor(target)
        out = {"data":img, "label":label}
        return out

    def make_balanced(self):
        balanced_no_beard = self.labels[self.labels["No_Beard"]==0].sample(n=2000, random_state=1)
        self.labels = self.labels.drop(self.labels[self.labels.No_Beard == 0].index)
        self.labels = pd.concat([balanced_no_beard, self.labels])

     
