import numpy as np
import matplotlib.pyplot as plt

import os
import json 

import torch
import torchvision.transforms as T

# Normalization values
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Denormalization transform
denormalize = T.Normalize(
    mean=[-m/s for m, s in zip(mean, std)],
    std=[1/s for s in std]
)

def check_images(step, batch_size, imgs, ground_truths, 
                 valid_probs, attrs, check_what, root_path="/content/upar_challenge/savefig/image_labels/mosque_camera"):

  save_number = 0
  for idx in range(imgs.size()[0]):
    attr_idx = attrs.index(check_what)
    if ground_truths[idx][attr_idx] == 1:

      image = denormalize(imgs[idx])
      image = image.cpu()
      # Clip the values to be in the range [0, 1]
      image = torch.clamp(image, 0, 1)
      # Convert the image to a numpy array and transpose the dimensions
      image = image.numpy().transpose((1, 2, 0))

      label_dict = dict(zip(attrs, valid_probs[idx].astype("str")))
      attr_path = os.path.join(root_path, check_what)
      if not os.path.exists(attr_path):
        os.makedirs(attr_path)

      image_index = batch_size * step + save_number
      image_path = os.path.join(attr_path, f"{image_index}.jpg")
      label_path = os.path.join(attr_path, f"{image_index}.json")

      save_number += 1
      plt.imshow(image)

      plt.savefig(image_path)
      json.dump(label_dict, open(label_path, 'w' ) )






