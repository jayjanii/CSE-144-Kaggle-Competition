import timm
import torch

from config import (
    MODEL_NAME, NUM_CLASSES, DEVICE,
    LR_HEAD_PHASE2, LR_BACKBONE_PHASE2,
    LR_HEAD_PHASE3, LR_BLOCKS_TOP_PHASE3, LR_BLOCKS_REST_PHASE3,
    TOP_BLOCKS_PHASE2,
)


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

def unfreeze_top_blocks(model: torch.nn.Module):
    """Phase 2: unfreeze top TOP_BLOCKS_PHASE2 blocks + head with differential LRs."""
    for param in model.parameters():
        param.requires_grad = False

    for param in model.head.parameters():
        param.requires_grad = True
    for param in model.blocks[-TOP_BLOCKS_PHASE2:].parameters():
        param.requires_grad = True

    return [
        {"params": model.head.parameters(),                      "lr": LR_HEAD_PHASE2},
        {"params": model.blocks[-TOP_BLOCKS_PHASE2:].parameters(), "lr": LR_BACKBONE_PHASE2},
    ]


def unfreeze_all(model: torch.nn.Module):
    """Phase 3: full model with differential LRs"""
    for param in model.parameters():
        param.requires_grad = True

    accounted = set(
        id(p)
        for p in list(model.head.parameters())
        + list(model.blocks[-TOP_BLOCKS_PHASE2:].parameters())
        + list(model.blocks[:-TOP_BLOCKS_PHASE2].parameters())
    )
    other_params = [p for p in model.parameters() if id(p) not in accounted]

    return [
        {"params": model.head.parameters(),                        "lr": LR_HEAD_PHASE3},
        {"params": model.blocks[-TOP_BLOCKS_PHASE2:].parameters(), "lr": LR_BLOCKS_TOP_PHASE3},
        {"params": model.blocks[:-TOP_BLOCKS_PHASE2].parameters(), "lr": LR_BLOCKS_REST_PHASE3},
        {"params": other_params,                                    "lr": LR_BLOCKS_REST_PHASE3},
    ]