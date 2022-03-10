import os
import uuid
import torch
import math
import matplotlib.pyplot as plt

def visualize(x, n=32, save_dir='/media/percy/1tb_ssd/features'):
    batch, channels, height, width = x.shape  # batch, channels, height, width
    if height > 1 and width > 1:
        f = os.path.join(save_dir, f"{uuid.uuid4().hex}.png")

        blocks = torch.chunk(x[0].cpu(), channels, dim=0)
        n = min(n, channels)  # number of plots
        fig, ax = plt.subplots(math.ceil(n / 8), 8, tight_layout=True)
        ax = ax.ravel()
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        for i in range(n):
            ax[i].imshow(blocks[i].squeeze().double().detach())
            ax[i].axis('off')

        print(f'Saving {f}... ({n}/{channels})')
        plt.savefig(f, dpi=300, bbox_inches='tight')
        plt.close()
