import torch.nn as nn

class CNN(nn.Module):

    def __init__(self, num_classes=3, channels=(32, 64, 128, 256), dropout=0.25):

        super().__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=channels[0]),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=channels[0]),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # (4, 32, 32, 32)
        
        self.block2 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=channels[1]),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels[1], channels[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=channels[1]),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # (4, 64, 16, 16)

        self.block3 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=channels[2]),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=channels[2]),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # (4, 128, 8, 8)

        self.block4 = nn.Sequential(
            nn.Conv2d(channels[2], channels[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=channels[3]),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels[3], channels[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=channels[3]),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)

        )
        # (4, 256, 4, 4)
        
        self.gap = nn.AdaptiveAvgPool2d((1,1))

        self.dropout = nn.Dropout(p=dropout)

        self.fc = nn.Linear(channels[3], num_classes)
        
    def forward_features(self, x):

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.gap(x)
        x = x.squeeze(-1).squeeze(-1)

        return x

    def forward(self, x):

        x = self.forward_features(x)
        x = self.dropout(x)
        x = self.fc(x)

        return x
    