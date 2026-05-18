import timm
import torch

from config import MODEL_NAME, NUM_CLASSES, DEVICE


def create_model(pretrained: bool = True) -> torch.nn.Module:
    model = timm.create_model(MODEL_NAME, pretrained=pretrained, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    print(f"Model {MODEL_NAME} loaded (pretrained={pretrained}).")
    return model


def freeze_backbone(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True


def unfreeze_all(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True
