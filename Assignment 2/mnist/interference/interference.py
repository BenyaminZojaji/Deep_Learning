import argparse
from tensorflow.keras.models import load_model
import numpy as np
import cv2

parser = argparse.ArgumentParser(description='Mnist interference by Benyamin Zojaji')
parser.add_argument('--input', type=str, help='image PATH', default='input.jpg')
parser.add_argument('--model', type=str, help='mlp/cnn', default='cnn')
args = parser.parse_args()

if args.model == 'mlp':
    model = load_model('mlp-mnist-2.h5')
else:
    model = load_model('CNN-MLP-mnist-optAdam.h5')

img = cv2.imread(args.input)
img = cv2.resize(img, (28,28))
img = cv2.cvtColor(img, (img, cv2.COLOR_BGR2GRAY))
img = img/255.0
img = img.reshape(1,28,28)

result = np.argmax(model.predict(img))