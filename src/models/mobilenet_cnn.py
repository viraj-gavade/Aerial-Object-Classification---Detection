import torch.nn as nn
import torchvision.models as models

def load_mobilenet(num_classes=2, pretrained=True, freeze=True):
    model = models.mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None)

    if freeze:
        for p in model.features.parameters():
            p.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, num_classes)
    )

    return model
