## MLP vs CNN+MLP


|               | Accuracy |                 |             |              |
| :-----------: | :------: | :-------------: | :---------: | :----------: |
| Dataset       | MLP      | MLP TF tutorial | CNN+MLP SGD | CNN+MLP Adam |
| Mnist         | 86.33%   | 97.45%          | 91.24%      | 99.03%       |
| Fashion Mnist | 75.98%   | 86.87%          | 71.66%      | 89.49%       |
| cifar 10      | 41.15%   | ---             | ---         | 70.07%       |
| cifar 100     | 20.66%   | ---             | ---         | 35.38%       |


```shell
usage: interfrence.py [--input INPUT] [--model MODEL]
```
