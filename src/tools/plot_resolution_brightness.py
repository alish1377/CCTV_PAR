import matplotlib.pyplot as plt
import numpy as np

def plot_resolution_brightness(normalized_resolution, normalized_brightness, save_path):
    """
    Plot the normalized resolution and brightness values.

    Parameters:
    - normalized_resolution: List of normalized resolution values.
    - normalized_brightness: List of normalized brightness values.
    """
    
    plt.figure(figsize=(10, 6))
    
    # Plotting the normalized resolution
    plt.subplot(2, 1, 1)
    plt.plot(normalized_resolution, label='Normalized Resolution', color='blue')
    plt.title('Normalized Resolution')
    plt.xlabel('Box Index')
    plt.ylabel('Normalized Value')
    plt.grid()
    
    # Plotting the normalized brightness
    plt.subplot(2, 1, 2)
    plt.plot(normalized_brightness, label='Normalized Brightness', color='orange')
    plt.title('Normalized Brightness')
    plt.xlabel('Box Index')
    plt.ylabel('Normalized Value')
    plt.grid()
    
    plt.tight_layout()
    #os.makedirs(save_path, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Figure saved to {save_path}")