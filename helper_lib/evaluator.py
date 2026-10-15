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


def get_predictions(model, data_loader, device="cpu"):
    """Return (predicted labels, true labels) for every sample in the loader."""
    model.to(device)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images.to(device))
            all_preds.append(outputs.argmax(dim=1).cpu())
            all_labels.append(labels)
    return torch.cat(all_preds), torch.cat(all_labels)


def confusion_matrix(preds, labels, num_classes=10):
    """matrix[i][j] = number of images of true class i predicted as class j."""
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(labels, preds):
        matrix[t, p] += 1
    return matrix


def per_class_accuracy(matrix):
    """Accuracy (%) for each true class: correct predictions / images of that class."""
    return (100 * matrix.diag().float() / matrix.sum(dim=1).float()).tolist()
