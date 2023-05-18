# EDD
Eye Disease Diagnosis


#### 1) Create a new environment

```bash
conda create -n classifer python=3.9 -y
```


#### 2) Please set your configuration.
```json
params = {
              "exp_name":"exp_3_load_epoch1iter80",
              "log_dir":"logs",
              "epochs": 5,
              "model_name": "Resnet50Pretrained",
              "device" : "mps",
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
```



#### Start training

```bash
python trainer.py
```