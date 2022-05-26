# %%
from typing import Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import torch


def draw(D: Union[torch.Tensor, np.ndarray], color: str = 'jet',
         dpi: int = 300, save: Optional[str] = None):
    plt.figure(dpi = dpi)
    plt.imshow(D, cmap = plt.get_cmap(color))
    if save is not None:
        plt.xticks([])
        plt.yticks([])
        plt.axis("off")
        plt.savefig(f"{save}", dpi = dpi)
    else:
        plt.colorbar()
    plt.show()
# %%
