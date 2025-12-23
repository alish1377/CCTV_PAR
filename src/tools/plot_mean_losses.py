import matplotlib.pyplot as plt

def plot_mean_losses(mean_upper_losses, mean_lower_losses, mean_rectified_upper_loses, mean_rectified_lower_loses, save_path):
    """
    Plot the mean losses for upper and lower branches.
    
    Parameters:
    - mean_upper_losses: List of mean upper losses.
    - mean_lower_losses: List of mean lower losses.
    - mean_rectified_upper_loses: List of mean rectified upper losses.
    - mean_rectified_lower_loses: List of mean rectified lower losses.
    """
    
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(mean_upper_losses, label='Mean Upper Losses', color='blue')
    plt.plot(mean_rectified_upper_loses, label='Mean Rectified Upper Losses', color='red')
    plt.title('Mean Upper Losses')
    plt.xlabel('Box Index') 
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(mean_lower_losses, label='Mean Lower Losses', color='green')
    plt.plot(mean_rectified_lower_loses, label='Mean Rectified Lower Losses', color='orange')
    plt.title('Mean Lower Losses')
    plt.xlabel('Box Index')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)