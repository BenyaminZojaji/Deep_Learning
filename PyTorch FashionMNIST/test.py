import argparse
import torch
import torchvision
from tqdm import tqdm
from model import MyModel

parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, default='weights.pth', help='weights PATH')
parser.add_argument('--processor', type=str, default='cpu', help='cpu/cuda')
parser.add_argument('--batchsize', type=int, default=64, help='batch size')
args = parser.parse_args()

def calc_acc(preds, labels):
  _, preds_max = torch.max(preds, 1)
  acc = torch.sum(preds_max == labels.data, dtype=torch.float64) / len(preds)
  return acc

def test_step(images, labels):
  global model, loss_function
  loss = 0.0
  acc = 0.0

  images = images.to(device)
  labels = labels.to(device)

  preds = model(images)

  loss = loss_function(preds, labels)

  acc = calc_acc(preds, labels)

  return loss, acc

def test(test_data_loader):
    test_loss = 0.0
    test_acc = 0.0
    for images, labels in tqdm(test_data_loader):
      t_loss, t_acc = test_step(images, labels)
      test_loss += t_loss
      test_acc += t_acc
    total_test_loss = test_loss / len(test_data_loader)
    total_test_acc = test_acc / len(test_data_loader)

    return total_test_acc, total_test_loss


batch_size = args.batchsize

transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0), (1))
])


if '__name__'=='__main__':
  dataset = torchvision.datasets.FashionMNIST('./dataset', train=False, download=True, transform=transform)
  test_data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

  device = torch.device(args.processor)

  model = MyModel()
  model.load_state_dict(torch.load(args.weights))
  model = model.to(device)

  model.eval()

  total_test_acc, total_test_loss = test(test_data_loader)
  print(f'Test Acc: {total_test_acc}, Test Loss: {total_test_loss}')