# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
from unets_torch import models

# %%
mps = torch.device("mps")

input_image_path = Path(
    "../images/images/20240822_pvdm_wt_1day_25_Straightened-6.tif"
)
checkpoint_path = Path("./checkpoints/best_model.pth")

# Load the model (make sure the parameters are the same as trained model)
model = models.AttentionUNet(
    1, 1, features=[32, 64, 128, 256], use_logits=True
)

checkpoint = torch.load(checkpoint_path, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])

model = model.to(mps)
model.eval()
# %%
# Load and preprocess the image
image = tifffile.imread(input_image_path).astype(np.float32)

# normalize image by percentile (same as training)
ilow, ihigh = np.percentile(image, (1.0, 99.0))
image = (image - ilow) / (ihigh - ilow)

# Convert to tensor and add batch dimension
image_tensor = torch.from_numpy(image)[None, None, ...].to(mps)

# Generate prediction
with torch.no_grad():
    logits = model(image_tensor)
    # Since use_logits=True, we need to apply sigmoid to get probabilities
    probabilities = torch.sigmoid(logits)

# Convert to numpy array
prob_map = probabilities.cpu().numpy()[0, 0]  # Remove batch and channel dims


# %% visualize result
fig, ax = plt.subplots(ncols=2, figsize=(10, 5), dpi=120)

ax[0].imshow(image, vmin=0, vmax=1)
ax[1].imshow(prob_map, cmap="gray")
ax[0].set_title("Input Image")
ax[1].set_title("Attention UNet, P(neuron|image)")
for a in ax:
    a.axis("off")

fig.savefig("test_eval_5epochs.png")
