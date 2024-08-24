
import wandb
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from abc import ABC, abstractmethod
import os 
import utils.util as util
import torch
import torchvision

class Log:
    def __init__(self, project_info : str, logger_name : str, is_active : bool, log_dir : str):
        self.project_info = project_info
        self.logger_name = logger_name
        self.is_active = is_active
        self.log_dir = log_dir
        self.logger = self.init_logger()
        

    def init_logger(self):
        if self.is_active:
            if self.logger_name == "wandb":
                return WandbLogger(self.project_info, self.log_dir, self.is_active)
            elif self.logger_name == "tensorboard":
                return TBLogger(self.project_info, self.log_dir, self.is_active)
            else:
                raise ValueError("Invalid logger type !!")
        else:
            return DefaultLogger()

        

class ILog(ABC):
    
    @abstractmethod
    def init(name):
        pass

    @abstractmethod
    def log_scaler(scalers : dict, step : int):
        pass
    
    def alert(self, text):
        # logging.warning(text)
        print("| Warnining |",text)

class DefaultLogger(ILog):
    def init(self, name):
        pass
    
    def log_scaler(self, scalers : dict, step: int):
        pass
    

class TBLogger(ILog):
    def __init__(self, project_info, log_dir:str, is_active : bool):
        self.is_active = is_active
        self.logdir = log_dir
        date = datetime.now(). strftime("%Y_%m_%d-%I:%M:%S_%p")
        self.log_dir_edited = f"{self.logdir}/tb_{date}"
        os.makedirs(self.log_dir_edited, exist_ok=True)
        util.writeJson(project_info, self.log_dir_edited+"/info.json" )
        self.init(self.log_dir_edited)
        # logging.info("Tensorboard logging is activated !!")

    def init(self, log_dir):
        if self.is_active:
            self.logger = SummaryWriter(log_dir)

    def log_scaler(self, scalers : dict, step : int):
        if self.is_active:
            for k, v in scalers.items():
                self.logger.add_scalar(k, v, step)

    def log_model(self, model, epoch, step, loss, acc):
        if self.is_active:
            models_dir = f"{self.log_dir_edited}/models"
            if os.path.exists(models_dir) == False:
                    os.makedirs(models_dir, exist_ok=True)

            torch.save(model.state_dict(), f"{models_dir}/net_best_epoch_{epoch}__iter_{step}__loss_{round(loss,4)}__acc_{acc}.pth")
            
    
    def log_image(self, batch, step, stage):
        if self.is_active:
            batch = torchvision.utils.make_grid(batch)
            self.logger.add_images(f"{stage}/images", batch, step, dataformats="CHW")
    
class WandbLogger(ILog):
    def __init__(self, project_info, log_dir : str, is_active : bool):
        self.is_active = is_active
        self.logdir = log_dir
        os.makedirs(self.logdir, exist_ok=True)
        self.init(project_info["name"], self.logdir)
        # logging.info("Wandb logging is activated !!")
        wandb.run.name = f"{project_info['archname']}_{wandb.run.id}"
        util.writeJson(project_info, wandb.run.dir+"/info.json" )


    def init(self, name, log_dir):
        if self.is_active:
            wandb.init(project=name, dir=log_dir)

    def log_scaler(self, scalers : dict, step : int):
        if self.is_active:
            wandb_dict = {}
            for k, v in scalers.items():
                wandb_dict[k] = scalers[k]
            wandb.log(wandb_dict, step=step)
    
    def log_model(self, model, epoch, step, loss):
        if self.is_active:
            raise NotImplementedError("Wandb logging is not implemented for model saving !!")
            # torch.save(model.state_dict(), f"{self.log_dir_edited}/net_best_epoch_{epoch}_iter_{step}_loss_{round(loss,4)}.pth")
       
