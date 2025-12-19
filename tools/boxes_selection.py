
import numpy as np
from collections import deque

# Dfine the function to select the best boxes based on the z-score of the normalized resolution and normalized brightness
def select_best_boxes(upper_losses, lower_losses, upper_accs, lower_accs,
                      normalized_resolution, normalized_brightness,
                      alpha=0.5, beta=0.5, selection_method="both_zscore"):
    
    rectified_upper_losses = []
    rectified_lower_losses = []
    rectified_upper_accs = []
    rectified_lower_accs = []
    brightness_zscores = []
    resolution_zscores = []


    resolution_buffer = deque(maxlen=30) 
    brightness_buffer = deque(maxlen=30) 



    number_of_boxes = len(upper_losses)
    for box_idx in range(number_of_boxes):


        if box_idx==0:
            pass
            #rectified_lower_losses.append(lower_losses[box_idx])
            #rectified_upper_losses.append(upper_losses[box_idx])

        if box_idx >= 15:
            resolution_buffer.append(normalized_resolution[box_idx])
            brightness_buffer.append(normalized_brightness[box_idx])
            brightness_mean = np.mean(brightness_buffer)
            brightness_std = np.std(brightness_buffer)

            brightness_zscore = (normalized_brightness[box_idx] - brightness_mean) / (brightness_std+0.000001)
            brightness_zscores.append(brightness_zscore)

            resolution_mean = np.mean(resolution_buffer)
            resolution_std = np.std(resolution_buffer)
            resolution_zscore = (normalized_resolution[box_idx] - resolution_mean) / (resolution_std+0.000001)
            resolution_zscores.append(resolution_zscore)

            if selection_method == "both_zscore":

                if brightness_zscore > 1.5 and not resolution_zscore < -1.5:
                    #print("salam")
                    # If the z-score is greater than 2, we consider it an outlier
                    rectified_lower_losses.append(lower_losses[box_idx])
                    rectified_upper_losses.append(upper_losses[box_idx])
                    rectified_lower_accs.append(lower_accs[box_idx])
                    rectified_upper_accs.append(upper_accs[box_idx])
            elif selection_method == "brightness_zscore":
                if brightness_zscore > 2:
                    # If the z-score is greater than 2, we consider it an outlier
                    rectified_lower_losses.append(lower_losses[box_idx])
                    rectified_upper_losses.append(upper_losses[box_idx])
                    rectified_lower_accs.append(lower_accs[box_idx])
                    rectified_upper_accs.append(upper_accs[box_idx])
            elif selection_method == "resolution_zscore":
                if resolution_zscore > 2:
                    # If the z-score is greater than 2, we consider it an outlier
                    rectified_lower_losses.append(lower_losses[box_idx])
                    rectified_upper_losses.append(upper_losses[box_idx])
                    rectified_lower_accs.append(lower_accs[box_idx])
                    rectified_upper_accs.append(upper_accs[box_idx])

    if len(rectified_upper_losses) == 0:
        #idx = brightness_zscores.index(max(brightness_zscores))
        box_idx = normalized_brightness.index(max(normalized_brightness))
        rectified_lower_losses.append(lower_losses[box_idx])
        rectified_upper_losses.append(upper_losses[box_idx])
        rectified_lower_accs.append(lower_accs[box_idx])
        rectified_upper_accs.append(upper_accs[box_idx])

    return brightness_zscores, resolution_zscores, rectified_lower_losses, rectified_upper_losses, rectified_lower_accs, rectified_upper_accs


        #if box_idx % 30 and box_idx % 30:
            
            #resolution_mean = np.mean(resolution_buffer)
            #resolution_std = np.std(resolution_buffer)

            # Calculate the z-scores
            #resolution_z_score = (normalized_resolution[box_idx] - resolution_mean) / resolution_std
            
                


