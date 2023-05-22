import torch.optim as optim
from torch import nn 
from model import get_model 
import torch
import numpy as np
import os
from logger import Log
from datetime import datetime
from loguru import logger as printer
from utils import EarlyStopper
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class TrainManager:
    def __init__(self, train_loader, test_loader, device, **kwargs):
        self.kwargs = kwargs
        self.exp_path = os.path.join(self.kwargs["log_dir"],self.kwargs["exp_name"])
        self.model = get_model(kwargs["model_name"], device, self.kwargs)

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        self.criterion = nn.CrossEntropyLoss()
        self.device = device
        self.early_stopper = EarlyStopper(patience=kwargs["patience"])

       
        self.best_loss = float("inf")
        self.log = Log(kwargs, kwargs["logger_name"],kwargs["logging_active"], self.exp_path)

        if self.kwargs["load_model"]:
            splits = self.kwargs["load_model_path"].split("__")
            load_epoch = int(splits[0].split("_")[-1])
            load_iter = int(splits[1].split("_")[-1])
            self.train_index = load_iter * load_epoch
            self.batch_start = load_iter
            self.epoch_start = load_epoch
        else:
            self.train_index = 0
            self.epoch_start = 1
            self.batch_start = 0

        self.test_index = 0
        

    def train(self):
        for epoch in range(self.epoch_start, self.kwargs["epochs"]+1):
            print("-----------------")
            print("# Train Epoch:", epoch)
            print("-----------------")
        
            for batch_idx, batch in enumerate(self.train_loader, self.batch_start+1):
                self.train_index += 1
                self.model.train()
                data, label = batch["data"].to(self.device), batch["label"].to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, label)
                loss.backward()
                self.optimizer.step()
                

                if self.train_index % self.kwargs["vis_print_per_iter"] == 0:
                    printer.info(f"Epoch| batch : {epoch} | {batch_idx} / {len(self.train_loader)} -> Train Loss: {loss.item()}" )
                    self.log.logger.log_scaler({"Train/CrossEntropyLoss": loss.item()}, self.train_index)
                    
                    pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                    acc = pred.eq(label.view_as(pred)).sum().item()/pred.shape[0]
                    self.log.logger.log_scaler({"Train/Accuracy": acc}, self.train_index)

                if self.train_index % self.kwargs["test_per_iter"] == 0:
                    val_loss, val_acc = self.test()

                    if self.early_stopper.early_stop(val_loss):
                        printer.warning("Early Stopping")
                        exit("Program stopped by Early Stopping")
                    
                    self.scheduler.step(val_loss)
                    if val_loss < self.best_loss and self.kwargs["logging_active"]:
                        self.best_loss = val_loss
                        self.log.logger.log_model(self.model, epoch, batch_idx, round(self.best_loss,4), val_acc)
                        printer.info("Model saved")
                        printer.info(f"Best Loss: {self.best_loss} | Accuracy: {val_acc}")


        print("End of the Training :)")   
        
    
    def test(self):
        printer.info("Testing ...")
        self.model.eval()
        test_loss = []
        test_acc = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                self.test_index += 1
                data, label = batch["data"].to(self.device), batch["label"].to(self.device)
                output = self.model(data)
                loss = self.criterion(output, label)
                test_loss.append(loss.item())

                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                acc = pred.eq(label.view_as(pred)).sum().item()/pred.shape[0]
                test_acc.append(acc)


                if self.test_index % self.kwargs["vis_print_per_iter"] == 0:
                    self.log.logger.log_scaler({"Test/CrossEntropyLoss": loss.item()}, self.test_index )
                    self.log.logger.log_scaler({"Test/Accuracy": acc}, self.train_index)
                
              
                
            
            
            printer.info(f"Mean Test Loss: {np.mean(test_loss)}")
            printer.info(f"Mean Test Accuracy:  {np.mean(test_acc)}")
        return np.mean(test_loss), np.mean(test_acc)
        
