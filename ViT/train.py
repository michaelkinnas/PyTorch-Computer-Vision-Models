import torch
import torch.nn as nn
from ViT import ViT
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim

from tqdm.auto import tqdm


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

batch_size = 4

trainset = CIFAR10(root='../../data', train=True,
                                        download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = CIFAR10(root='../../data', train=False,
                                       download=True, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


model = ViT(img_size=32, num_channels=3, patch_size=4, embedding_dim=512, n_heads=4, n_transformer_layers=4, num_classes=10)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

EPOCHS = 20

epoch_iterator = tqdm(range(EPOCHS), position=0, desc='Epoch', leave=False)
for epoch in epoch_iterator:
    batch_iterator = tqdm(trainloader, position=1, desc='Batch', leave=False)

    batch_running_loss = 0
    batch_running_acc = 0

    for i, (X, y) in enumerate(batch_iterator):
        
        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        loss.backward()

        optimizer.zero_grad()
        optimizer.step()

        batch_running_loss += loss.item()

        batch_iterator.set_postfix_str(f"Train loss: {batch_running_loss / (i+1):.3f}")
    

    test_loss = 0
    test_acc = 0

    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(testloader):       
            y_pred = model(X)

            test_loss += loss_fn(y_pred, y).item()
            test_acc += (y_pred == y).sum().item()

    model.train()
    
    epoch_iterator.set_postfix_str(f"Test loss: {test_loss / len(testloader):.3f} | Test accuracy: {test_acc / len(testloader):.3f}")

