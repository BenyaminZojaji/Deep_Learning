import argparse
import os
import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm
import cv2
import pandas as pd
import numpy as np
from model import Model
from torch.utils.data import DataLoader
from dataset import Dataset

parser = argparse.ArgumentParser()

parser.add_argument('--processor', type=str, default='cpu', help='cpu/cuda')
parser.add_argument('--batchsize', type=int, default=64, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning-rate')
parser.add_argument('--epoch', type=int, default=10, help='number of epoch')
parser.add_argument('--path', type=str, default='/content/utkface_aligned_cropped/crop_part1', help='path of dataset')
args = parser.parse_args()


def train_step(images, labels):
    global device, optimizer, loss_function
    loss = 0.0

    images = images.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()

    # 1- forwarding
    preds = model(images)

    # 2- backwarding
    loss = loss_function(preds, labels)
    loss.backward()

    # 3- Update
    optimizer.step()

    return loss

def train(epochs):
  for epoch in range(epochs):
    train_loss= 0.0

    for images, labels in tqdm(train_data_loader):
      t_loss = train_step(images, labels)
      train_loss += t_loss
    total_train_loss = train_loss / len(train_data_loader)

    print(f'Epoch:{epoch}, Train Loss: {total_train_loss}')


transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

if '__name__'=='__main__':
    batch_size = args.batchsize

    images = [] # X
    ages = [] # Y
    width = height = 224

    for image_name in os.listdir(args.path):
        ages.append(int(image_name.split('_')[0]))

        image = cv2.imread(f'{args.path}/{image_name}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (width, height))
        images.append(image)
    images = pd.Series(images, name='Images')
    ages = pd.Series(ages, name='Ages')
    dataframe = pd.concat([images, ages], axis=1)
    X = np.array(dataframe['Images'].values.tolist())
    Y = np.array(dataframe['Ages'].values.tolist())
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    X = torch.permute(X, (0, 3, 2, 1))

    dataset = Dataset(X, Y, transform)
    train_data_loader = DataLoader(dataset, batch_size=batch_size)

    device = torch.device(args.processor)
    
    model = Model()
    model = model.to(device)
    model.train(True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_function = torch.nn.CrossEntropyLoss()


    train(args.epochs)