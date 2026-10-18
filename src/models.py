import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


class MLP(nn.Module):
    """Course baseline: 64x64 gray -> 128 ReLU -> one logit."""
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((64, 64))
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(4096, 128), nn.ReLU(),
                                 nn.Dropout(.3), nn.Linear(128, 1))
        nn.init.kaiming_normal_(self.net[1].weight, nonlinearity='relu')
        nn.init.zeros_(self.net[1].bias)
        nn.init.xavier_uniform_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(self.pool(x.mean(dim=1, keepdim=True)))


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for a, b in [(3, 16), (16, 32), (32, 64)]:
            layers += [nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b),
                       nn.ReLU(), nn.MaxPool2d(2)]
        self.net = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                 nn.Dropout(.3), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x)


def build_model(name, pretrained=True):
    if name == 'mlp':
        return MLP()
    if name == 'small_cnn':
        return SmallCNN()
    if name == 'resnet18':
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        model.fc = nn.Sequential(nn.Dropout(.3), nn.Linear(model.fc.in_features, 1))
        return model
    raise ValueError(name)


def configure_stage(model, name, stage):
    for p in model.parameters():
        p.requires_grad = True
    if name == 'resnet18':
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True
        if stage == 'finetune':
            for p in model.layer4.parameters():
                p.requires_grad = True


def training_mode(model, name):
    model.train()
    if name == 'resnet18':
        # Keep pretrained BN statistics fixed, including during layer4 fine-tuning.
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
