import argparse
from tensorflow.keras.models import load_model
import numpy as np
import cv2

parser = argparse.ArgumentParser(description='cifar10 interference by Benyamin Zojaji')
parser.add_argument('--input', type=str, help='image PATH', default='input.jpg')
parser.add_argument('--model', type=str, help='mlp/cnn', default='cnn')
args = parser.parse_args()

if args.model == 'mlp':
    model = load_model('mlp-cfar10.h5')
else:
    model = load_model('cnn-mlp-cfar10.h5')

img = cv2.imread(args.input)
img = cv2.resize(img, (32,32))
img = cv2.cvtColor(img, (img, cv2.COLOR_BGR2GRAY))
img = img/255.0
img = img.reshape(32,32,3)

result = np.argmax(model.predict(img))
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
print(class_names[result])
