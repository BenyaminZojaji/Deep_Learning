# Deep Learning
## Assignment 1
### Face Classification
- Face Classification with **DeepFace** library.
- Read 200 images from 4 different actors, use `ArcFace` model to get Feature vectors of them and save in a csv file.
- Create MLP with tensorflow and fit data.

| Algorithm | MLP |
| --------- |:---:|
| Accuracy | 80% |

---
## Assignment 2
### MLP vs CNN+MLP

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

### Sheikh Detector
- The database consists of 4 categories of images. you can view it <a href='#'>here</a>.
- Using VGG16 model.
- Classes:
  - Normal people 👨🏻
  - Sheikh 👳🏻‍♂️

</br>

|  VGG16               | Loss               | Accuracy          |
| :------------------: | :----------------: | :---------------: |
| Train                |  0.16              |   92.45%          |
| Validation           |  0.19              |   96.15%          |
| Test                 |  0.28              |   95.83%          |

</br>

- <a href='http://t.me/isHeSheikhbot'>A telegram-bot that classify people by sheikh or not.</a>

</br>

- Confusion Matrix:
  > ![confmat](https://user-images.githubusercontent.com/77120507/158569528-8188f50f-c95f-452d-9f2d-b16e09c4783d.png)
---
### Face Mask Detection
- Face Mask Detection using **Tensorflow Keras**, **PySide6**, **open-cv**.
- Dataset: <a href='https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset'>Images Dataset</a>
- Model:
  - MobileNetV2

</br>

|                      | Accuracy          | Loss              |
| :------------------: | :----------------:| :---------------: |
| MobileNetV2          |   99.29%          |  0.02             |

</br>


> https://user-images.githubusercontent.com/77120507/159115385-75fb3494-471c-46bd-aa86-f13935adc3ed.MP4

---
### 17 Flowers
- Dataset: <a href='https://www.kaggle.com/datasets/saidakbarp/17-category-flowers'>17 category flowers</a>
- 17 flowers Classification using Tensorflow Keras.
- Model:
  - Resnet50V2
  - Xception
  - InceptionResNetV2


</br>

|                      | Accuracy          | Loss              |
| :------------------: | :----------------:| :---------------: |
| ResNet50V2           |   84.19%          |  0.51             |
| Xception             |   81.99%          |  0.61             |
| InceptionResNetV2    |   75.37%          |  0.75             |

</br>

---
### Houses Price
- Dataset: <a href='https://github.com/emanhamed/Houses-dataset'>Link</a> 
- usage:

  ````shell
  pip install -r requirements.txt
  ````
  ````shell
  python cnn_regression.py -d HousesDataset
  ````

- inference:
  - Simply add 4 image into `pic` folder, including bathroom, bedroom, kitchen and frontal of house.
  - usage:
   
     ````shell
     python inference.py
     ````
  
---
### Age Estimation
- Dataset: <a href='https://www.kaggle.com/datasets/jangedoo/utkface-new'>utkface</a>
- Estimating human age using Tensorflow Keras.
- Model:
  - Xception
  - Resnet50V2

|           | Loss(mse) |
| --------- |:---------:|
| Xception  | 124.68    |
| Resnet50V2| 145.41    |

---
### Face Recognition
- Face Recognition exercise using Tensorflow.


|  | Accuracy | Loss |
|:-:|:-:|:-:|
| Model | 84% | 0.02 |

- Inference:

  ````shell
  usage: python inference.py [image PATH] [weight PATH]
  ````
