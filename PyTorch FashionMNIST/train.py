import argparse
import torch
import torchvision
from tqdm import tqdm
from model import MyModel

parser = argparse.ArgumentParser()

parser.add_argument('--processor', type=str, default='cpu', help='cpu/cuda')
parser.add_argument('--batchsize', type=int, default=64, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning-rate')
parser.add_argument('--epoch', type=int, default=10, help='number of epoch')
args = parser.parse_args()

def calc_acc(preds, labels):
  _, preds_max = torch.max(preds, 1)
  acc = torch.sum(preds_max == labels.data, dtype=torch.float64) / len(preds)
  return acc

def train_step(images, labels):
    global device, optimizer, loss_function
    loss = 0.0
    acc = 0.0

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

    acc = calc_acc(preds, labels)

    return loss, acc

def train(epochs):
  for epoch in range(epochs):
    train_loss= 0.0
    train_acc = 0.0

    for images, labels in tqdm(train_data_loader):
      t_loss, t_acc = train_step(images, labels)
      train_loss += t_loss
      train_acc += t_acc
    total_train_loss = train_loss / len(train_data_loader)
    total_train_acc = train_acc / len(train_data_loader)
    print(f'Train Acc: {total_train_acc}, Train Loss: {total_train_loss}')


transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0), (1))
])

if '__name__'=='__main__':
    batch_size = args.batchsize
    dataset = torchvision.datasets.FashionMNIST('./dataset', train=True, download=True, transform=transform)
    train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device(args.processor)
    
    model = MyModel()
    model = model.to(device)
    model.train(True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_function = torch.nn.CrossEntropyLoss()


    train(args.epochs)
