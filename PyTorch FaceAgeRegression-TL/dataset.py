from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class Dataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.data = X
        self.target = y
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        if self.transform:
            x = self.transform(x)

        return x, y
    
    def __len__(self):
        return len(self.data)

transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

def build_dataset(X_train, Y_train, batch_size):
  dataset = Dataset(X_train, Y_train, transform)
  train_data_loader = DataLoader(dataset, batch_size=batch_size)
  return train_data_loader
