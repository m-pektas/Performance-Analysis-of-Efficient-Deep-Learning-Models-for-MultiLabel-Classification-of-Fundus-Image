

- Epoch = 100
- bs = 15
- patience = 20
- lr = 0.001
- LRSchedular = ReduceLROnPlateau
- TransferLearning = Imagenet


| Method  | Train Split | Best Test Loss |  Best Epoch | Augmentation | Accuracy  |
|---|---|---|---|---|---|
|  Resnext50_32x4d | 0.7  | 0.6570  | 12 | - |  85.09 % |
|  Resnet50  | 0.7  | 0.6815   |  12 | - |83.73 %|
|  EfficientNetB3 | 0.7  | 0.4297  | 10 | - | 90.14 % |
|  EfficientNetB3 | 0.7  | 0.4691  | 7 | RandomEqualize(0.5) | 86.70 % |
|  SwinTiny | 0.7  | 1.5682  | 1 | - | 43.47 % |

