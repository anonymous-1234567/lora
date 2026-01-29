import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_features(model, loader, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            # Forward pass through the backbone
            out = model(x)
            # Flatten if necessary (ResNet outputs usually [B, 512, 1, 1] -> [B, 512])
            out = out.view(out.size(0), -1) 
            
            features.append(out.cpu())
            labels.append(y)
            
    return torch.cat(features), torch.cat(labels)

# Setup
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 1. Load Pre-trained ResNet18
resnet = torchvision.models.resnet18(pretrained=True)

# 2. Strip the classification head (fc layer)
# ResNet18 structure: ... -> avgpool -> fc
# We want the output of avgpool.
# A common trick is to replace 'fc' with Identity, 
# strictly speaking, we want the output of the layer BEFORE fc.
resnet.fc = nn.Identity() 
resnet.to(device)

# 3. Data Loaders (CIFAR10)
# Note: ResNet expects 224x224 usually, but can work on 32x32. 
# For best results with ImageNet weights, upsample to 224 or use a CIFAR-specific ResNet.
transform = transforms.Compose([
    transforms.Resize(224), # Resize to match ImageNet training size
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) #mean and std of ImageNet
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=128, shuffle=False, num_workers=0)
testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)

# 4. Extract and Save
print("Extracting training features...")
X_train, y_train = get_features(resnet, trainloader, device)
print("Extracting test features...")
X_test, y_test = get_features(resnet, testloader, device)

# Save to disk to avoid re-computing
torch.save((X_train, y_train, X_test, y_test), 'cifar10_resnet18_features.pt')