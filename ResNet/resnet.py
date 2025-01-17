import torch.nn.functional as fn
import torch.nn as nn


__all__ = ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']

class _Block(nn.Module):
    def __init__(self, config): 
        super().__init__()

        self.seq = nn.Sequential()

        for i, (inc, outc, ks, pad) in enumerate(config):
            self.seq.append(nn.Conv2d(in_channels=inc, 
                                      out_channels=outc, 
                                      kernel_size=ks, 
                                      stride=(2 if i == 0 and config[0][0] * 2 == config[-1][1] else 1), 
                                      padding=pad))
            self.seq.append(nn.BatchNorm2d(outc))
            self.seq.append(nn.ReLU())
        
        self.seq.pop(-1) #remove last relu from sequence
        
        #TODO Maybe refactor these logic checks?
        if config[0][0] * 2 == config[-1][1]:
            self.projection = nn.Conv2d(in_channels=config[0][0], 
                                        out_channels=config[-1][1], 
                                        kernel_size=1, 
                                        stride=2)
        elif config[0][0] < config[-1][1]:
            self.projection = nn.Conv2d(in_channels=config[0][0], 
                                        out_channels=config[-1][1], 
                                        kernel_size=1, 
                                        stride=1)
        else:
            self.projection = nn.Identity()

    def forward(self, x):
        residual = self.projection(x)
        x = self.seq(x)
        x += residual
        return fn.relu(x)


class _Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq = nn.Sequential()

        for block in config:
            self.seq.append(_Block(config=block))

    def forward(self, x):
        return self.seq(x)
    

class _Net(nn.Module):
    def __init__(self, config, in_channels, out_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=64, 
                               kernel_size=7, 
                               stride=2, 
                               padding=2)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, 
                                    stride=2, 
                                    padding=1)

        self.seq = nn.Sequential()
        for layer in config:
            self.seq.append(_Layer(config=layer))
    
        self.avgpool = nn.AvgPool2d(kernel_size=7, 
                                    stride=1)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=config[-1][-1][-1][1], 
                            out_features=out_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.seq(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x



def ResNet18(in_channels=3, out_classes=1000):
    from .configs import resnet18
    return _Net(config=resnet18, in_channels=in_channels, out_classes=out_classes)
 
def ResNet34(in_channels=3, out_classes=1000):
    from .configs import resnet34
    return _Net(config=resnet34, in_channels=in_channels, out_classes=out_classes)

def ResNet50(in_channels=3, out_classes=1000):
    from .configs import resnet50
    return _Net(config=resnet50, in_channels=in_channels, out_classes=out_classes)

def ResNet101(in_channels=3, out_classes=1000):
    from .configs import resnet101
    return _Net(config=resnet101, in_channels=in_channels, out_classes=out_classes)

def ResNet152(in_channels=3, out_classes=1000):
    from .configs import resnet152
    return _Net(config=resnet152, in_channels=in_channels, out_classes=out_classes)