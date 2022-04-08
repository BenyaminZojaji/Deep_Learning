import argparse
import time
import torch
import torchvision
import cv2
import numpy as np
from model import MyModel

parser = argparse.ArgumentParser()

parser.add_argument('--processor', type=str, default='cpu', help='cpu/cuda')
parser.add_argument('--input', type=str, default='input.jpg', help='input PATH')
parser.add_argument('--weights', type=str, default='weights.pth', help='weights PATH')
args = parser.parse_args()

transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0), (1))
])

device = torch.device(args.processor)

model = MyModel()
model.load_state_dict(torch.load(args.weights))
model = model.to(device)

model.eval()

# oreoricess
img = cv2.imread('/content/pullover.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (28, 28))
tensor = transform(img).unsqueeze(0).to(device)

# process
start = time.time()
preds = model(tensor)
end = time.time()

# postprocess
# _, output=torch.max(preds, 1)
preds = preds.cpu().detach().numpy()
output = np.argmax(preds)
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(f'{classes[output]}, Process-Time: {end-start}')