# import torch
import torch.nn as nn

class Conv3Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        return x
        

class Conv4Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2,  padding=1)

        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class VGG16(nn.Module):
    def __init__(self, in_channels=3, dropout=0.2, output_classes = 10):
        super().__init__()

        self.block_3_1 = Conv3Block(in_channels=in_channels, out_channels = 64)
        self.block_3_2 = Conv3Block(in_channels=64, out_channels=128)

        self.block_4_1 = Conv3Block(in_channels=128, out_channels=256)

        self.block_4_2 = Conv4Block(in_channels=256, out_channels=512)
        self.block_4_3 = Conv4Block(in_channels=512, out_channels=512) #This also outputs 512 channels

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=25088, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=4096, out_features=output_classes)
        )

        self.sf = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.block_3_1(x)
        x = self.block_3_2(x)
        x = self.block_4_1(x)
        x = self.block_4_2(x)
        x = self.block_4_3(x)
        x = self.fc(x)
        x = self.sf(x)
        return x
