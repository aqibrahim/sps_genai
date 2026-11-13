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


def train_diffusion(model, data_loader, criterion=None, optimizer=None, device="cpu", epochs=10,
                    ema_decay=0.999, on_epoch_end=None):
    """Train a DiffusionModel with the DDPM noise-prediction objective (Module 8 activity).

    Each batch: sample one random timestep t per image, add noise in closed form to get x_t,
    predict the noise with the U-Net, and minimize criterion(predicted noise, true noise).
    criterion defaults to MSE; optimizer defaults to AdamW(lr=1e-3, weight_decay=1e-4).

    An exponential moving average (EMA) of the U-Net weights is kept, since it gives smoother,
    higher-quality samples. After training, model.ema_unet holds the EMA copy (used for
    generation) and model.history the per-epoch losses. Returns the model.
    on_epoch_end(epoch, ema_unet) can save sample grids.
    """
    import copy

    model.to(device)
    criterion = criterion or torch.nn.MSELoss()
    optimizer = optimizer or torch.optim.AdamW(model.unet.parameters(), lr=1e-3, weight_decay=1e-4)
    ema_unet = copy.deepcopy(model.unet).eval()
    for p in ema_unet.parameters():
        p.requires_grad_(False)
    history = {"loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total, batches = 0.0, 0
        for x0, _ in data_loader:
            x0 = x0.to(device)
            predicted_noise, noise = model.noise_prediction(x0)
            loss = criterion(predicted_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():  # EMA update: ema = decay * ema + (1 - decay) * current
                for p_ema, p in zip(ema_unet.parameters(), model.unet.parameters()):
                    p_ema.mul_(ema_decay).add_(p.detach(), alpha=1 - ema_decay)
                for b_ema, b in zip(ema_unet.buffers(), model.unet.buffers()):
                    b_ema.copy_(b)
            total += loss.item()
            batches += 1
        history["loss"].append(total / batches)
        print(f"Epoch {epoch}/{epochs} - noise-prediction loss: {history['loss'][-1]:.4f}")
        if on_epoch_end is not None:
            on_epoch_end(epoch, ema_unet)

    model.ema_unet = ema_unet
    model.history = history
    return model


def train_ebm(model, data_loader, device="cpu", epochs=10, lr=1e-4, alpha=0.1,
              langevin_steps=60, step_size=10.0, noise_std=0.005, buffer_size=8192,
              reinit_prob=0.05, on_epoch_end=None):
    """Train an energy-based model with contrastive divergence and Langevin sampling.

    Each batch:
      1. Negative samples: start from a replay buffer of earlier samples (95%) or fresh uniform
         noise (5%), and run short-run Langevin dynamics on the *images* (model weights frozen).
      2. Loss (contrastive divergence + regularization):
             L = mean E(x_real) - mean E(x_fake) + alpha * (mean E(x_real)^2 + mean E(x_fake)^2)
         The first terms push energy down on real images and up on model samples; the
         regularizer keeps the energy values from drifting to large magnitudes.
      3. Update the model weights with Adam; store the new samples in the buffer.
    Returns a history dict. on_epoch_end(epoch, model, buffer) can save sample grids.
    """
    from helper_lib.generator import langevin_sample

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.0, 0.999))
    buffer = torch.rand(buffer_size, 3, 32, 32, device=device) * 2 - 1  # uniform noise in [-1, 1]
    history = {"loss": [], "cd_loss": [], "energy_real": [], "energy_fake": []}

    for epoch in range(1, epochs + 1):
        sums = {k: 0.0 for k in history}
        batches = 0
        for real, _ in data_loader:
            real = real.to(device)
            b = real.size(0)
            # 1) Negative samples: mostly continue earlier chains, sometimes restart from noise
            idx = torch.randint(0, buffer_size, (b,), device=device)
            start = buffer[idx]
            fresh = torch.rand(b, device=device) < reinit_prob
            start[fresh] = torch.rand_like(start[fresh]) * 2 - 1
            fake = langevin_sample(model, start, steps=langevin_steps, step_size=step_size, noise_std=noise_std)
            buffer[idx] = fake

            # 2) Contrastive divergence loss: gradients flow to the weights here, not to the images
            model.train()
            real = real + 0.005 * torch.randn_like(real)  # small noise on real data stabilizes training
            e_real, e_fake = model(real), model(fake)
            cd_loss = e_real.mean() - e_fake.mean()
            reg = alpha * (e_real.pow(2).mean() + e_fake.pow(2).mean())
            loss = cd_loss + reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sums["loss"] += loss.item()
            sums["cd_loss"] += cd_loss.item()
            sums["energy_real"] += e_real.mean().item()
            sums["energy_fake"] += e_fake.mean().item()
            batches += 1

        for k in history:
            history[k].append(sums[k] / batches)
        print(f"Epoch {epoch}/{epochs} - loss: {history['loss'][-1]:.4f} - CD: {history['cd_loss'][-1]:.4f} - "
              f"E(real): {history['energy_real'][-1]:.4f} - E(fake): {history['energy_fake'][-1]:.4f}")
        if on_epoch_end is not None:
            on_epoch_end(epoch, model, buffer)
    return history


def train_lstm(model, data_loader, criterion, optimizer, device="cpu", epochs=10, val_loader=None, clip=1.0):
    """Train a next-word language model: cross-entropy between the predicted distribution at every
    position and the actual next word. Gradients are clipped (common for RNNs, avoids exploding
    gradients). Returns the model; model.history holds per-epoch loss and perplexity."""
    import math

    model.to(device)
    history = {"train_loss": [], "train_perplexity": [], "val_loss": [], "val_perplexity": []}
    for epoch in range(1, epochs + 1):
        model.train()
        total, count = 0.0, 0
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            total += loss.item() * x.size(0)
            count += x.size(0)
        history["train_loss"].append(total / count)
        history["train_perplexity"].append(math.exp(total / count))
        msg = f"Epoch {epoch}/{epochs} - loss: {history['train_loss'][-1]:.4f} - perplexity: {history['train_perplexity'][-1]:.1f}"
        if val_loader is not None:
            model.eval()
            v_total, v_count = 0.0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits, _ = model(x)
                    v_total += criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item() * x.size(0)
                    v_count += x.size(0)
            history["val_loss"].append(v_total / v_count)
            history["val_perplexity"].append(math.exp(v_total / v_count))
            msg += f" - val loss: {history['val_loss'][-1]:.4f} - val perplexity: {history['val_perplexity'][-1]:.1f}"
        print(msg)
    model.history = history
    return model
