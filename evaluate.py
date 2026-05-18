import torch


@torch.inference_mode()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        predictions = outputs.argmax(dim=1)

        running_loss += loss.item() * inputs.size(0)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total
