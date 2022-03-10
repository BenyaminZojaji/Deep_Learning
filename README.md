# Deep Learning
## Assignment 1
#### Face Classification
- Face Classification with **DeepFace** library.
- Read 200 images from 4 different actors, use `ArcFace` model to get Feature vectors of them and save in a csv file.
- Create MLP with tensorflow and fit data.

| Algorithm | MLP |
| --------- |:---:|
| Accuracy | 80% |

---
## Assignment 2
#### MLP vs CNN+MLP

- <a href="https://drive.google.com/drive/folders/1gBmgFu9Unbmq9jR4XSnrO7h6h2YxihpW?usp=sharing">Download Models</a>.

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
---

