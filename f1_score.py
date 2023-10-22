
from loader import ODIR5K
from torch.utils.data import DataLoader
import numpy as np
import torch
import random
import os
from loguru import logger as printer
from model import get_model
import argparse

class Evaluater:

    def __init__(self, params):
        self.params = params
        self.set_seed(42)

        self.device = self.params.get("device")

       
        test_dataset = ODIR5K(self.params.get("img_dir"),
                                     self.params.get("label_dir"),
                                     train_test_size=self.params.get("train_test_size"), 
                                     is_train=False, augment=self.params.get("augment"))
        
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

        model = get_model(self.params.get("model_name"), self.device, {})
        model.load_state_dict(torch.load(self.params.get("load_model_path")))
        model.eval()
        true_class = []
        pred_class = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                data, label = batch["data"].to(self.device), batch["label"].to(self.device)
                output = model(data)
               
            

                pred = output.argmax(dim=1, keepdim=True)
                true_class += list(label)
                pred_class += list(pred)
        


    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='F1')
    parser.add_argument('--model_name', default="EfficientNetB1Pretrained", help='You can use all models in model.py')
    parser.add_argument('--img_dir', default="data/preprocessed_images", help='data_dir')
    parser.add_argument('--label_dir', default="data/full_df.csv", help='labels')
    parser.add_argument('--batch_size', default=15, help='bs')
    parser.add_argument('--shuffle', default=True, help='shuffle images')
    parser.add_argument('--train_test_size', default=0.8, help='iter_count')
    parser.add_argument('--load_model', default=True, help='device')
    parser.add_argument('--load_model_path', default="logs/EDD_Seed_and_LRSchedular_Exp/tb_2023_05_29-12:59:34_AM/models/net_best_epoch_9__iter_133__loss_0.4826__acc_0.8796875000000001.pth", help='warmup')
    parser.add_argument('--num_workers', default=4, help='iter_count')
    parser.add_argument('--device', default="cuda", help='device')
    args = parser.parse_args()
    params = vars(args)
    print(args)

    trainer = Evaluater(params)
    trainer.run()