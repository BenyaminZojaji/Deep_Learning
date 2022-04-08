import torch

class MyModel(torch.nn.Module):
  def __init__(self):
    super().__init__()

    self.fc1 = torch.nn.Linear(28*28, 128)
    self.fc2 = torch.nn.Linear(128, 256)
    self.fc3 = torch.nn.Linear(256, 512)
    self.fc4 = torch.nn.Linear(512, 10)

  def forward(self, x):
    # input shape: 32 * 28 * 28 * 1 <-> batch-size * height * width * channel
    x = x.reshape((x.shape[0], 784))
    # output shape: 32 * 784

    x = self.fc1(x)
    x = torch.relu(x)
    x = torch.dropout(x, 0.3, train=True)
    x = self.fc2(x)
    x = torch.relu(x)
    x = torch.dropout(x, 0.3, train=True)
    x = self.fc3(x)
    x = torch.relu(x)
    x = torch.dropout(x, 0.3, train=True)
    x = self.fc4(x)
    x = torch.softmax(x, dim=1)
    return x