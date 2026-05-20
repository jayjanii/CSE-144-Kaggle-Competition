import torch


class EarlyStopper:
    """Signals when a phase should stop due to val_acc plateauing.

    Tracks the best seen value and increments a patience counter each epoch
    that fails to improve by at least min_delta.  Call it after each
    validation step; it returns True when training should stop.

    Args:
        patience:  epochs without improvement before stopping.
        min_delta: minimum improvement over the current best to reset the
                   counter.  Filters out noise from near-flat regions.
    """

    def __init__(self, patience: int, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -float("inf")
        self.counter = 0

    def __call__(self, val_acc: float) -> bool:
        if val_acc > self.best + self.min_delta:
            self.best = val_acc
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler: torch.amp.GradScaler, mixup_fn=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        if mixup_fn is not None:
            inputs, labels = mixup_fn(inputs, labels)

        optimizer.zero_grad()
        with torch.amp.autocast(device):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds = outputs.argmax(dim=1)
        total_loss += loss.item() * inputs.size(0)
        # hard labels unavailable when mixup active — skip accuracy tracking
        if mixup_fn is None:
            correct += (preds == labels).sum().item()
        total += inputs.size(0)

    acc = correct / total if mixup_fn is None else float("nan")
    return total_loss / total, acc


def train_one_epoch_phase2(model, loader, criterion, optimizer, device, accum_steps, scaler: torch.amp.GradScaler, mixup_fn=None):
    model.train()
    optimizer.zero_grad()

    total_loss, total_correct, total_samples = 0.0, 0, 0

    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)

        with torch.amp.autocast(device):
            outputs = model(images)
            loss = criterion(outputs, labels) / accum_steps

        scaler.scale(loss).backward()

        if (i + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps * images.size(0)
        preds = outputs.argmax(dim=1)
        if mixup_fn is None:
            total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    # flush any leftover gradients from the final partial accumulation window
    if (len(loader)) % accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    acc = total_correct / total_samples if mixup_fn is None else float("nan")
    return total_loss / total_samples, acc
