"""Training loop."""
import torch


def train_model(model, data_loader, criterion, optimizer, device="cpu", epochs=10):
    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()  # training mode (matters for dropout / batch norm)
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        print(f"Epoch {epoch}/{epochs} - loss: {running_loss / total:.4f} - "
              f"train acc: {100 * correct / total:.2f}%")
    return model
