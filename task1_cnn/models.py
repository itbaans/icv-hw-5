import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelA(nn.Module):
    """
    Model A - Baseline CNN
    3 conv blocks (Conv2D -> BatchNorm -> ReLU -> MaxPool)
    Filter progression: 32 -> 64 -> 128
    Global Average Pooling
    Dense -> Softmax
    Total parameters <= 1.5M
    """
    def __init__(self):
        super(ModelA, self).__init__()
        # Input shape: (3, 128, 128)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Global Average Pooling will reduce (128, H, W) to (128, 1, 1) -> 128
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 1)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # Note: CrossEntropyLoss applies Softmax internally, so we don't return softmax output here
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ModelB(nn.Module):
    """
    Model B - Deeper/Regularized CNN
    At least 4 conv blocks with dropout (0.2-0.5), L2 weight decay, optional residual connection.
    """
    def __init__(self, dropout_rate=0.3):
        super(ModelB, self).__init__()
        self.in_channels = 32
        
        # Initial Conv
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Residual Blocks
        self.layer1 = self._make_layer(32, stride=1)
        self.layer2 = self._make_layer(64, stride=2)
        self.layer3 = self._make_layer(128, stride=2)
        self.layer4 = self._make_layer(256, stride=2)
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(256, 1)
        
    def _make_layer(self, out_channels, stride):
        strides = [stride]
        layers = []
        for s in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def get_activation(name):
    if name == 'leaky_relu':
        return nn.LeakyReLU(0.1, inplace=True)
    return nn.ReLU(inplace=True)

class SimpleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu'):
        super(SimpleBlock, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = get_activation(activation)
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        return self.pool(self.relu(self.bn(self.conv(x))))

class ResidualBlockAblation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu'):
        super(ResidualBlockAblation, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2, 2)
        self.act = get_activation(activation)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act(out)
        return self.pool(out)

class AblationModel(nn.Module):
    """
    Dynamic CNN for ablation study.
    Allows changing number of layers, filters per layer, kernel size, and residual connections.
    """
    def __init__(self, filters=[32, 64, 128], kernel_size=3, dropout_rate=0.3, use_residual=False, activation='relu'):
        super(AblationModel, self).__init__()
        
        layers = []
        in_channels = 3
        
        for out_channels in filters:
            if use_residual:
                layers.append(ResidualBlockAblation(in_channels, out_channels, kernel_size, activation))
            else:
                layers.append(SimpleBlock(in_channels, out_channels, kernel_size, activation))
            in_channels = out_channels
            
        self.features = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        
        # Output is regression (1 continuous value)
        self.fc = nn.Linear(filters[-1], 1)
        
    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
