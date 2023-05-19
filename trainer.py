
from loader import ODIR5K
from torch.utils.data import DataLoader
from train import TrainManager
import numpy as np
import torch
import random
import os
from loguru import logger as printer

class Trainer:

    def __init__(self, params):
        self.params = params
        self.set_seed(42)

        if self.params["logging_active"]:
            printer.add(os.path.join(self.params["log_dir"],self.params["exp_name"])+"/{time}.log")

        train_dataset = ODIR5K(self.params.get("img_dir"),
                                     self.params.get("label_dir"),
                                     train_test_size=self.params.get("train_test_size"), 
                                     is_train=True)
        
        test_dataset = ODIR5K(self.params.get("img_dir"),
                                     self.params.get("label_dir"),
                                     train_test_size=self.params.get("train_test_size"), 
                                     is_train=False)
        
        self.train_loader = DataLoader(dataset=train_dataset,
                              batch_size=self.params["batch_size"],
                              shuffle=self.params["shuffle"],
                              num_workers=self.params["num_workers"])
        self.test_loader = DataLoader(dataset=test_dataset,
                              batch_size=self.params["batch_size"],
                              shuffle=self.params["shuffle"],
                              num_workers=self.params["num_workers"])
    def set_seed(self, seed: int = 42) -> None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set a fixed value for the hash seed
        os.environ["PYTHONHASHSEED"] = str(seed)
        printer.info(f"Random seed set as {seed}")


    def run(self):
        train_manager = TrainManager(train_loader=self.train_loader,
                                     test_loader=self.test_loader, 
                                     **self.params)
                                    
        train_manager.train()
        



    
if __name__ == "__main__":

    params = {
              "exp_name":"EDD_Seed_and_LRSchedular_Exp",
              "img_dir": "data/preprocessed_images",
              "label_dir" : "data/full_df.csv",
              "log_dir":"logs",
              "epochs": 100,
              "model_name": "Resnext50_32x4dPretrained",
              "device" : "cuda",
              "batch_size": 15,
              "shuffle": True,
              "patience" : 10,
              "num_workers": 4,
              "logger_name": "tensorboard",
              "logging_active": True,
              "vis_print_per_iter": 5,
              "test_per_iter": 50,
              "train_test_size": 0.7,
              "load_model": False,
              "load_model_path": "logs/exp_2/tb_2022_11_11-03:19:29_PM/models/net_best_epoch_1__iter_80__loss_1.3364__acc_0.45.pth"}

    
    trainer = Trainer(params)
    trainer.run()