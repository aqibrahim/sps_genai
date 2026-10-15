"""Evaluation metrics."""
import torch


def evaluate_model(model, data_loader, criterion, device="cpu"):
    model.to(device)
    model.eval()  # inference mode (disables dropout, uses batch-norm running stats)
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    print(f"Test loss: {avg_loss:.4f} - Test accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy
