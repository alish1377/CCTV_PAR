import numpy as np
import matplotlib.pyplot as plt

import os
import json 

import torch
import torchvision.transforms as T

from random import randint

# Normalization values
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Denormalization transform
denormalize = T.Normalize(
    mean=[-m/s for m, s in zip(mean, std)],
    std=[1/s for s in std]
)


colors_rgb = {
    "black": (0,0,0),
    "blue": (0,0,255),
    "brown": (165,42,42),
    "green": (0,255,0),
    "grey": (128,128,128),
    "orange": (255,165,0),
    "pink": (255,192,203),
    "purple": (128,0,128),
    "red": (255,0,0),
    "white": (255,255,255),
    "yellow": (255,255,0),
    #"others": (randint(0, 255), randint(0, 255), randint(0, 255))
}


def extract_color(label_dict):
    u_rgb = ()
    l_rgb = ()

    # Ensure the order of the scores matches the order of the colors in the dictionary
    color_names = list(colors_rgb.keys())

    # get upperbody and lowerbody color tuples
    upper_color_lists = list(label_dict.values())[8:19]
    upper_color_lists = list(map(float, upper_color_lists))
    lower_color_lists = list(label_dict.values())[20:31]
    lower_color_lists = list(map(float, lower_color_lists))

    u_score, l_score = sum(upper_color_lists),sum(lower_color_lists)
    u_weighted_r, u_weighted_g, u_weighted_b = 0,0,0
    l_weighted_r, l_weighted_g, l_weighted_b = 0,0,0

    for i in range(11):
        r,g,b = colors_rgb[color_names[i]]

        u_weighted_r += r*upper_color_lists[i]
        u_weighted_g += g*upper_color_lists[i]
        u_weighted_b += b*upper_color_lists[i]

        l_weighted_r += r*lower_color_lists[i]
        l_weighted_g += g*lower_color_lists[i]
        l_weighted_b += b*lower_color_lists[i]

    # Calculate the weighted average of RGB components
    u_average_r = u_weighted_r / u_score
    u_average_g = u_weighted_g / u_score
    u_average_b = u_weighted_b / u_score

    l_average_r = l_weighted_r / l_score
    l_average_g = l_weighted_g / l_score
    l_average_b = l_weighted_b / l_score

    # The resulting color
    u_rgb = (round(u_average_r), round(u_average_g), round(u_average_b))
    l_rgb = (round(l_average_r), round(l_average_g), round(l_average_b))

    return u_rgb, l_rgb

def save_upper_lower_colors(step, batch_size, imgs, ground_truths, 
                 valid_probs, attrs, root_path="savefig/color_mixture/mosque_camera"):

  save_number = 0
  for idx in range(imgs.size()[0]):

    image = denormalize(imgs[idx])
    image = image.cpu()
    # Clip the values to be in the range [0, 1]
    image = torch.clamp(image, 0, 1)
    # Convert the image to a numpy array and transpose the dimensions
    image = image.numpy().transpose((1, 2, 0))

    label_dict = dict(zip(attrs, valid_probs[idx].astype("str")))
    u_rgb, l_rgb = extract_color(label_dict)

    image_index = batch_size * step + save_number
    index_path = os.path.join(root_path, f"{image_index}")
    if not os.path.exists(index_path):
        os.makedirs(index_path)

    image_path = os.path.join(index_path, f"{image_index}.jpg")
    upperlower_path = os.path.join(index_path, f"ul_{image_index}.jpg")

    save_number += 1

    plt.imshow(image)
    plt.savefig(image_path)

    # Convert RGB values from 0-255 range to 0-1 range for matplotlib
    u_rgb_normalized = tuple(c / 255 for c in u_rgb)
    l_rgb_normalized = tuple(c / 255 for c in l_rgb)

    # Create a figure and a set of subplots
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the first color
    ax[0].add_patch(plt.Rectangle((0, 0), 1, 1, color=u_rgb_normalized))
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(0, 1)
    ax[0].axis('off')  # Hide the axes
    ax[0].set_title('Upperbody')

    # Plot the second color
    ax[1].add_patch(plt.Rectangle((0, 0), 1, 1, color=l_rgb_normalized))
    ax[1].set_xlim(0, 1)
    ax[1].set_ylim(0, 1)
    ax[1].axis('off')  # Hide the axes
    ax[1].set_title('Lowerbody')

    # Display the plot
    plt.savefig(upperlower_path, bbox_inches='tight')
    plt.show()


      