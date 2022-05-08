import os
import argparse
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
parser.add_argument('--weights', type=str, default='weights.pth', help='weights PATH')
parser.add_argument('--processor', type=str, default='cpu', help='cpu/cuda')
parser.add_argument('--batchsize', type=int, default=64, help='batch size')
parser.add_argument('--path', type=str, default='/content/utkface_aligned_cropped/crop_part1', help='path of dataset')
args = parser.parse_args()


def test_step(images, labels):
    global model, loss_function
    loss = 0.0

    images = images.to(device)
    labels = labels.to(device)

    preds = model(images)

    loss = loss_function(preds, labels)

    return loss

def test(test_data_loader):
    test_loss = 0.0
    for images, labels in tqdm(test_data_loader):
        t_loss = test_step(images, labels)
        test_loss += t_loss
    total_test_loss = test_loss / len(test_data_loader)


    return total_test_loss


batch_size = args.batchsize


transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


if '__name__'=='__main__':
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
    test_data_loader = DataLoader(dataset, batch_size=batch_size)

    device = torch.device(args.processor)

    model = Model()
    model.load_state_dict(torch.load(args.weights))
    model = model.to(device)

    model.eval()

    total_test_loss = test(test_data_loader)
    print(f'Test Loss: {total_test_loss}')