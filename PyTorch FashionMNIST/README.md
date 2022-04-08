## Fashion MNIST
- Fashion MNIST classification with `PyTorch`.
- Dataset: <a href='https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html#fashionmnist'>PyTorch Dataset</a>


|| Accuracy | Loss |
|:-:|:-:|:-:|
| Personal Model | 84% | 1.62 |

- Inference:
  ```shell
  python inference.py --processor[Optional def:cpu cpu/cuda] --weights[weights path] --input[input image path]
  ```
  
- Train:
   ```shell
   python train.py --processor[def:cpu cpu/cuda] --batchsize[def:64] --lr[learning-rate] --epoch[number of epochs]
   ```
