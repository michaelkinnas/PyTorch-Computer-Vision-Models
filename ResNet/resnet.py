import torch.nn as nn
import torch.nn.functional as fn

#A single conv layer for resnet
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, padding=0, stride=1, relu=True):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        if relu: self.relu = nn.ReLU(inplace=False)
    
    def forward(self, x):
        x = self.bn(self.conv(x))
        if hasattr(self, "relu"):
            x = self.relu(x)
        return x
       
class ConvBlock2(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv2_1_1 = ConvLayer(in_channels=64, out_channels=64, padding=1)
        self.conv2_1_2 = ConvLayer(in_channels=64, out_channels=64, padding=1, relu=False)

        self.conv2_2_1 = ConvLayer(in_channels=64, out_channels=64, padding=1)
        self.conv2_2_2 = ConvLayer(in_channels=64, out_channels=64, padding=1, relu=False)

        self.conv2_3_1 = ConvLayer(in_channels=64, out_channels=64, padding=1)
        self.conv2_3_2 = ConvLayer(in_channels=64, out_channels=64, padding=1, relu=False)


    def forward(self, x):
        res = x.clone()

        x = self.conv2_1_1(x)
        x = self.conv2_1_2(x)
        x += res
        x = fn.relu(x)

        res = x.clone()

        x = self.conv2_2_1(x)
        x = self.conv2_2_2(x)
        x += res
        x = fn.relu(x)

        res = x.clone()

        x = self.conv2_3_1(x)
        x = self.conv2_3_2(x)
        x += res
        x = fn.relu(x)

        return x
        

class ConvBlock3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3_1_1 = ConvLayer(in_channels=64, out_channels=128, padding=1, stride=2)
        self.conv3_1_2 = ConvLayer(in_channels=128, out_channels=128, padding=1, relu=False)

        self.conv3_2_1 = ConvLayer(in_channels=128, out_channels=128, padding=1)
        self.conv3_2_2 = ConvLayer(in_channels=128, out_channels=128, padding=1, relu=False)

        self.conv3_3_1 = ConvLayer(in_channels=128, out_channels=128, padding=1)
        self.conv3_3_2 = ConvLayer(in_channels=128, out_channels=128, padding=1, relu=False)

        self.conv3_4_1 = ConvLayer(in_channels=128, out_channels=128, padding=1)
        self.conv3_4_2 = ConvLayer(in_channels=128, out_channels=128, padding=1, relu=False)

        self.resize_projection = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2)

    def forward(self, x):
        res = x.clone()

        x = self.conv3_1_1(x) 
        x = self.conv3_1_2(x)
        x += self.resize_projection(res)
        x = fn.relu(x)

        res = x.clone()

        x = self.conv3_2_1(x)
        x = self.conv3_2_2(x)
        x += res
        x = fn.relu(x)

        res = x.clone()

        x = self.conv3_3_1(x)
        x = self.conv3_3_2(x)
        x += res
        x = fn.relu(x)

        res = x.clone()

        x = self.conv3_4_1(x)
        x = self.conv3_4_2(x)
        x += res
        x = fn.relu(x)

        return x


class ConvBlock4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv4_1_1 = ConvLayer(in_channels=128, out_channels=256, padding=1, stride=2)
        self.conv4_1_2 = ConvLayer(in_channels=256, out_channels=256, padding=1, relu=False)

        self.conv4_2_1 = ConvLayer(in_channels=256, out_channels=256, padding=1)
        self.conv4_2_2 = ConvLayer(in_channels=256, out_channels=256, padding=1, relu=False)

        self.conv4_3_1 = ConvLayer(in_channels=256, out_channels=256, padding=1)
        self.conv4_3_2 = ConvLayer(in_channels=256, out_channels=256, padding=1, relu=False)

        self.conv4_4_1 = ConvLayer(in_channels=256, out_channels=256, padding=1)
        self.conv4_4_2 = ConvLayer(in_channels=256, out_channels=256, padding=1, relu=False)

        self.conv4_5_1 = ConvLayer(in_channels=256, out_channels=256, padding=1)
        self.conv4_5_2 = ConvLayer(in_channels=256, out_channels=256, padding=1, relu=False)

        self.conv4_6_1 = ConvLayer(in_channels=256, out_channels=256, padding=1)
        self.conv4_6_2 = ConvLayer(in_channels=256, out_channels=256, padding=1, relu=False)

        self.resize_projection = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2)

    def forward(self, x):
        res = x.clone()

        x = self.conv4_1_1(x)
        x = self.conv4_1_2(x)
        x += self.resize_projection(res)
        x = fn.relu(x)

        res = x.clone()    

        x = self.conv4_2_1(x)
        x = self.conv4_2_2(x)
        x += res
        x = fn.relu(x)

        res = x.clone()

        x = self.conv4_3_1(x)
        x = self.conv4_3_2(x)
        x += res
        x = fn.relu(x)

        res = x.clone()

        x = self.conv4_4_1(x)
        x = self.conv4_4_2(x)
        x += res
        x = fn.relu(x)

        res = x.clone()

        x = self.conv4_5_1(x)
        x = self.conv4_5_2(x)
        x += res
        x = fn.relu(x)

        res = x.clone()

        x = self.conv4_6_1(x)
        x = self.conv4_6_2(x)
        x += res
        x = fn.relu(x)
        
        return x
    

class ConvBlock5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv5_1_1 = ConvLayer(in_channels=256, out_channels=512, padding=1, stride=2)
        self.conv5_1_2 = ConvLayer(in_channels=512, out_channels=512, padding=1, relu=False)

        self.conv5_2_1 = ConvLayer(in_channels=512, out_channels=512, padding=1)
        self.conv5_2_2 = ConvLayer(in_channels=512, out_channels=512, padding=1, relu=False)

        self.conv5_3_1 = ConvLayer(in_channels=512, out_channels=512, padding=1)
        self.conv5_3_2 = ConvLayer(in_channels=512, out_channels=512, padding=1, relu=False)

        self.resize_projection = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=2)


    def forward(self, x):
        res = x.clone()
        
        x = self.conv5_1_1(x)
        x = self.conv5_1_2(x)
        x += self.resize_projection(res)
        x = fn.relu(x)

        res = x.clone()

        x = self.conv5_2_1(x)
        x = self.conv5_2_2(x)
        x += res
        x = fn.relu(x)

        res = x.clone()

        x = self.conv5_3_1(x)
        x = self.conv5_3_2(x)
        x += res
        x = fn.relu(x)
        
        return x



class ResNet34(nn.Module):
    def __init__(self, in_channels=3, out_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=2)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block2 = ConvBlock2()

        self.block3 = ConvBlock3()

        self.block4 = ConvBlock4()

        self.block5 = ConvBlock5()

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=512, out_features=out_classes)
    
    def forward(self, x):
        x = self.conv1(x)

        x = self.maxpool(x)

        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x
        #keep this out and transform it to next dimensions for next block
        