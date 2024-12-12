"""
debugging script to inspect tensor sizes

"""

# %%
import torch
import torch.nn as nn
from unets_torch import models


# %%
class Hook:
    def __init__(self, module, backward=False):
        if isinstance(module, nn.ModuleList):
            module = module
        else:
            module = [module]
        self.module = module
        for sub_module in self.module:
            if not backward:
                self.hook = sub_module.register_forward_hook(self.hook_fn)
            else:
                self.hook = sub_module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


# %%
mps = torch.device("mps")
model = models.UNet(1, 1, features=[32, 64, 128, 256])
model = model.to(mps)

forward_hooks = [Hook(layer[1]) for layer in list(model.named_modules())]

sample_input = torch.randn(1, 1, 800, 800, dtype=torch.float32).to(mps)
output = model(sample_input)

for hook in forward_hooks:
    if isinstance(hook.input, torch.Tensor):
        input_shape = hook.input.shape
    else:
        input_shape = hook.input[0].shape
    if isinstance(hook.output, torch.Tensor):
        output_shape = hook.output.shape
    else:
        output_shape = hook.output[0].shape

    print("Input shape:\t", input_shape)
    print("Output shape:\t", output_shape)
    print("---" * 17)
