import argparse
import numpy as np
import cv2
from model import MyModel

parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, default='faceRecognition_weights', help='weights PATH')
parser.add_argument('--image', type=str, default='input.jpg', help='image PATH')
args = parser.parse_args()

model = MyModel()
model.load_weights(args.weights)

img = cv2.imread(args.input_image)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))
img = img / 255.0
img = img[np.newaxis, ...]

person = ['Ali Khamenei','Angelina Jolie','Barak Obama','Behnam Bani','Donald Trump','Emma Watson','Han Hye Jin','Kim Jong Un','Leyla Hatami','Lionel Messi','Michelle Obama','Morgan Freeman','Queen Elizabeth','Scarlett Johansson']
result = np.argmax(model.predict(img))
print(person[result])