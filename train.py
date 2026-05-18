import torch
from torch.cuda.amp import autocast, GradScaler


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        predictions = outputs.argmax(dim=1)
        running_loss += loss.item() * inputs.size(0)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def train_one_epoch_phase2(model, loader, criterion, optimizer, device, accum_steps, scaler: GradScaler):
    model.train()
    optimizer.zero_grad()

    total_loss, total_correct, total_samples = 0.0, 0, 0

    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels) / accum_steps

        scaler.scale(loss).backward()

        if (i + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps  # undo the division for logging
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    # flush any leftover gradients from the final partial accumulation window
    if (len(loader)) % accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return total_loss / len(loader), total_correct / total_samples
