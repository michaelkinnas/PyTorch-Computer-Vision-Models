import torch.nn as nn


class _Block(nn.Module):
    def __init__(self, config, batch_norm):
        super().__init__()

        self.seq = nn.Sequential()

        for inc, outc, ks, st, pad in config:
            self.seq.append(nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=ks, stride=st, padding=pad))
            if batch_norm:
                self.seq.append(nn.BatchNorm2d(num_features=config[-1][1]))
            self.seq.append(nn.ReLU(inplace=False))
            
        self.seq.append(nn.MaxPool2d(kernel_size=2, stride=2))       
    
    def forward(self, x):
        return self.seq(x)


# Minimum image size 32
class _Net(nn.Module):
    def __init__(self, config, dropout, hidden_layer_size, batch_norm, out_classes, image_size):
        super().__init__()

        hidden_in_layer_size = (image_size // 2 ** 5) ** 2 * config[-1][-1][1]
        assert hidden_in_layer_size >= 1, "Image size is too small for this network depth"

        self.seq = nn.Sequential()

        for layer in config:
            self.seq.append(_Block(config=layer, batch_norm=batch_norm))       
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_in_layer_size, out_features=hidden_layer_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(in_features=hidden_layer_size, out_features=hidden_layer_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(in_features=hidden_layer_size, out_features=out_classes)
        )

    
    def forward(self, x):
        x = self.seq(x)
        print(x.shape)
        x = self.fc(x)
        return x



def VGG11A(out_classes=1000, image_size=224, hidden_layer_size=4096, dropout=0.5, batch_norm = False):
    from .configs import vgg11A
    return _Net(config=vgg11A, out_classes=out_classes, hidden_layer_size=hidden_layer_size, dropout=dropout,image_size=image_size, batch_norm=batch_norm)

def VGG13B(out_classes=1000, image_size=224, hidden_layer_size=4096, dropout=0.5, batch_norm = False):
    from .configs import vgg13B
    return _Net(config=vgg13B, out_classes=out_classes, hidden_layer_size=hidden_layer_size, dropout=dropout,image_size=image_size, batch_norm=batch_norm)

def VGG16C(out_classes=1000, image_size=224, hidden_layer_size=4096, dropout=0.5, batch_norm = False):
    from .configs import vgg16C
    return _Net(config=vgg16C, out_classes=out_classes, hidden_layer_size=hidden_layer_size, dropout=dropout,image_size=image_size, batch_norm=batch_norm)

def VGG16D(out_classes=1000, image_size=224, hidden_layer_size=4096, dropout=0.5, batch_norm = False):
    from .configs import vgg16D
    return _Net(config=vgg16D, out_classes=out_classes, hidden_layer_size=hidden_layer_size, dropout=dropout,image_size=image_size, batch_norm=batch_norm)

def VGG19E(out_classes=1000, image_size=224, hidden_layer_size=4096, dropout=0.5, batch_norm = False):
    from .configs import vgg19E
    return _Net(config=vgg19E, out_classes=out_classes, hidden_layer_size=hidden_layer_size, dropout=dropout, image_size=image_size, batch_norm=batch_norm)