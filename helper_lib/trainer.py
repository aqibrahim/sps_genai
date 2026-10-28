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


def train_gan(gan, data_loader, device="cpu", epochs=20, lr=2e-4, betas=(0.5, 0.999),
              fixed_noise=None, on_epoch_end=None):
    """Train a GAN with the standard alternating updates.

    Each batch:
      1. Discriminator step: maximize log D(x) + log(1 - D(G(z)))
         (BCE with label 1 for real images and 0 for generated images).
      2. Generator step (non-saturating loss): maximize log D(G(z))
         (BCE with label 1 for generated images), through the discriminator.

    Returns a history dict with per-epoch average losses and discriminator outputs.
    on_epoch_end(epoch, generator) is called after each epoch (e.g. to save sample grids).
    """
    G, D = gan.generator.to(device), gan.discriminator.to(device)
    criterion = torch.nn.BCELoss()
    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=betas)
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=betas)
    history = {"d_loss": [], "g_loss": [], "d_real": [], "d_fake": []}

    for epoch in range(1, epochs + 1):
        G.train()
        D.train()
        sums = {k: 0.0 for k in history}
        batches = 0
        for real, _ in data_loader:
            real = real.to(device)
            b = real.size(0)
            ones = torch.ones(b, 1, device=device)
            zeros = torch.zeros(b, 1, device=device)

            # 1) Discriminator: real -> 1, fake -> 0 (fake detached so G is not updated)
            z = torch.randn(b, G.latent_dim, device=device)
            fake = G(z)
            d_real = D(real)
            d_fake = D(fake.detach())
            d_loss = criterion(d_real, ones) + criterion(d_fake, zeros)
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            # 2) Generator: make D label the fakes as real
            g_loss = criterion(D(fake), ones)
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

            sums["d_loss"] += d_loss.item()
            sums["g_loss"] += g_loss.item()
            sums["d_real"] += d_real.mean().item()
            sums["d_fake"] += d_fake.mean().item()
            batches += 1

        for k in history:
            history[k].append(sums[k] / batches)
        print(f"Epoch {epoch}/{epochs} - D loss: {history['d_loss'][-1]:.4f} - "
              f"G loss: {history['g_loss'][-1]:.4f} - D(x): {history['d_real'][-1]:.3f} - "
              f"D(G(z)): {history['d_fake'][-1]:.3f}")
        if on_epoch_end is not None:
            on_epoch_end(epoch, G)
    return history
