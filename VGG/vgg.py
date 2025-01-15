# import torch
import torch.nn as nn

class Conv3Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=False)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

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

        self.relu = nn.ReLU(inplace=False)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2,  padding=0)        

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
    def __init__(self, in_channels=3, dropout=0.5, output_classes = 10, image_size=224):
        super().__init__()
        self.block_3_1 = Conv3Block(in_channels=in_channels, out_channels = 64)
        self.block_3_2 = Conv3Block(in_channels=64, out_channels=128)

        self.block_4_1 = Conv3Block(in_channels=128, out_channels=256)
        self.block_4_2 = Conv4Block(in_channels=256, out_channels=512)
        self.block_4_3 = Conv4Block(in_channels=512, out_channels=512) #This also outputs 512 channels

        hidden_layer_size = self.__calculate_hidden_layer_size(image_size=image_size)
        assert hidden_layer_size >= 1, "Image size is too small for this network depth"

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=int(hidden_layer_size), out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(in_features=4096, out_features=output_classes)
        )

        self.sf = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.block_3_1(x) #16
        x = self.block_3_2(x) # 8
        x = self.block_4_1(x) # 4
        x = self.block_4_2(x) # 2
        x = self.block_4_3(x) # 1
        x = self.fc(x)
        # x = self.sf(x) #TODO Test if this is usually used from inside the model
        return x

    def __calculate_hidden_layer_size(self, image_size):
        return image_size ** 2 * 512 / 4 ** 5

