# %%
import matplotlib.pyplot as plt
import tifffile

from data_helpers import MaskDataset, create_dataloaders

get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")

# %%

train_loader, val_loader = create_dataloaders(
    image_dir="../images/images/",
    mask_dir="../images/masks/",
    batch_size=8,
    num_workers=0,
)

# %%
imgs, msks, fns = next(iter(train_loader))

# %%
fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
ax[0].imshow(imgs[7, 0].numpy(), vmin=0, vmax=1)
ax[1].imshow(msks[7, 0].numpy())
