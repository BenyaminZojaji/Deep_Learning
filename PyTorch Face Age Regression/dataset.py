from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self, X, y, transform=None): # I could write transform in getitem
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
