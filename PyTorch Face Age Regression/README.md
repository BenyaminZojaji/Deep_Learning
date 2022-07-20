## Face Age Regression

- Dataset: <a href='https://www.kaggle.com/datasets/jangedoo/utkface-new'>utkface</a>


|| MSE |
|:-:|:-:|
| Personal Model | 568.52 |

- Inference:
  ```shell
  python inference.py --processor[default:cpu cpu/cuda] --weights[weights path] --input[input image path]
  ```
  
- Train:
   ```shell
   python train.py --processor[default:cpu cpu/cuda] --batchsize[default:64] --lr[learning-rate] --epoch[default:10 number of epochs] --path[path of dataset]
   ```
- Test:
  ```shell
  python test.py --weights[weights path] --processor[default:cpu cpu/cuda] --batchsize[default:64] --path[path of dataset]
  ```

