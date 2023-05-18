
from loader import CelebHQLoader
from torch.utils.data import DataLoader
from train import TrainManager

class Trainer:

    def __init__(self, params):
        self.params = params

        train_dataset = CelebHQLoader(self.params.get("img_dir"),
                                     self.params.get("label_dir"),
                                     train_test_size=self.params.get("train_test_size"), 
                                     is_train=True)
        
        test_dataset = CelebHQLoader(self.params.get("img_dir"),
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
                
    def run(self):
        train_manager = TrainManager(train_loader=self.train_loader,
                                     test_loader=self.test_loader, 
                                     **self.params)
                                    
        train_manager.train()
        



    
if __name__ == "__main__":

    params = {
              "exp_name":"exp_5binaryfixed",
              "img_dir": "data/Celaba_StyleganAligned",
              "label_dir" : "data/combined_annotation.txt",
              "log_dir":"logs",
              "epochs": 50,
              "model_name": "MobileNetV2Pretrained",
              "device" : "cuda",
              "batch_size": 20,
              "shuffle": True,
              "num_workers": 4,
              "logger_name": "tensorboard",
              "logging_active": True,
              "vis_print_per_iter": 5,
              "test_per_iter": 20,
              "train_test_size": 0.7,
              "load_model": False,
              "load_model_path": "logs/exp_2/tb_2022_11_11-03:19:29_PM/models/net_best_epoch_1__iter_80__loss_1.3364__acc_0.45.pth"}

    trainer = Trainer(params)
    trainer.run()