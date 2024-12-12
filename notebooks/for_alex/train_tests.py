# %%
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from unets_torch import models

from data_helpers import create_dataloaders

# %% setup data loaders
train_loader, val_loader = create_dataloaders(
    image_dir="../images/images/",
    mask_dir="../images/masks/",
    batch_size=2,
    num_workers=0,
)

# %%
device = torch.device("mps")
model = models.AttentionUNet(
    1, 1, features=[32, 64, 128, 256], use_logits=True
)
model = model.to(device)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=5, factor=0.5
)

save_dir = Path("checkpoints")
save_dir.mkdir(exist_ok=True)

best_val_loss = float("inf")

# %%
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    train_losses = []

    train_progress = tqdm(
        enumerate(train_loader),
        desc=f"Epoch {epoch+1}/{num_epochs}",
        total=len(train_loader),
    )

    for step, (images, masks, _) in train_progress:
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, masks)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        avg_loss = sum(train_losses) / len(train_losses)
        train_progress.set_description(f"avg train_loss : {avg_loss:.4f}")

    # Validation phase
    model.eval()
    val_losses = []
    val_progress = tqdm(
        enumerate(val_loader), desc="Validation", total=len(val_loader)
    )

    with torch.no_grad():
        for step, (images, masks, _) in val_progress:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, masks)
            val_losses.append(loss.item())
            avg_val_loss = sum(val_losses) / len(val_losses)
            val_progress.set_description(f"avg val_loss : {avg_val_loss:.4f}")

    # Calculate epoch metrics
    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)

    # Log metrics
    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")

    # Update learning rate
    scheduler.step(val_loss)

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            },
            save_dir / "best_model.pth",
        )
