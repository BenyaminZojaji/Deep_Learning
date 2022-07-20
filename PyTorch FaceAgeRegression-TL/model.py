import torch
import torchvision
import torch.nn as nn

def build_model():
  model = torchvision.models.resnet50(pretrained=True)
  model.fc = nn.Linear(model.fc.in_features, 1)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = model.to(device)

  ct = 0
  for child in model.children():
    ct += 1
    if ct < 7:
      for param in child.parameters():
        param.requires_grad = False
  return model