import matplotlib.pyplot as plt
import os
import numpy as np

def plot_zscore_resolution_brightness(resolution_zscores, brightness_zscores, save_path):
    """
    Plots the z-scores of resolution and brightness

    Parameters:
    - resolution_zscores: List of z-scores for resolution.
    - brightness_zscores: List of z-scores for brightness.
    """
    
    plt.figure(figsize=(10, 6))
    
    # Plotting the z-scores for brightness
    plt.subplot(2, 1, 1)
    plt.plot(resolution_zscores, label='Mean BrigResolutionhtness Z-Scores', color='blue')
    plt.title('Z-Scores of Resolution')
    plt.xlabel('Box Index')
    plt.ylabel('Z-Score')
    plt.grid()
    
    # Plotting the z-scores for resolution
    plt.subplot(2, 1, 2)
    plt.plot(brightness_zscores, label='Mean Brightness Z-Scores', color='orange')
    plt.title('Z-Scores of Brightness')
    plt.xlabel('Box Index')
    plt.ylabel('Z-Score')
    plt.grid()
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Figure saved to {save_path}")