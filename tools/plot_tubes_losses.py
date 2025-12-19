import matplotlib.pyplot as plt
import os
import numpy

def plot_resolution_brightness_vs_loss(image_resolutions, image_brightness, upper_losses, lower_losses, save_path):
    """
    Plots the effect of image resolution and brightness on model losses.

    Parameters:
    - image_resolutions: List of total pixels per image (width * height)
    - image_brightness: List of average brightness values (0_1 scale)
    - upper_losses: List of loss values for the upper branch
    - lower_losses: List of loss values for the lower branch
    """

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ---- Plot 1: Loss vs. Resolution ----
    axes[0].plot(image_resolutions, upper_losses, 'o', linestyle='', label='Upper BodyLoss')
    axes[0].plot(image_resolutions, lower_losses, '*', linestyle='', label='Lower Body Loss')
    axes[0].set_title("Loss vs. Resolution")
    axes[0].set_xlabel("Image Resolution (width * height)")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # ---- Plot 2: Loss vs. Brightness ----
    axes[1].plot(image_brightness, upper_losses, 'o', label='Upper Body Loss')
    axes[1].plot(image_brightness, lower_losses, '*', label='Lower Body Loss')
    axes[1].set_title("Loss vs. Brightness (V)")
    axes[1].set_xlabel("Average Brightness (V channel)")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    #os.makedirs(save_path, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Figure saved to {save_path}")
    #plt.show()
