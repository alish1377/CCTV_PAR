import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def save_hists(pos_accs, neg_accs, all_accs, attrs):

  # Create the corresponding dictionaries
  pos_acc_dict = dict(zip(attrs, pos_accs))
  neg_acc_dict = dict(zip(attrs, neg_accs))
  all_accs_dict = dict(zip(attrs, all_accs))

  # All labels
  bar_save(all_accs_dict, "attr_acc", "plots/attr_acc.png")
  # Positive labels
  bar_save(pos_acc_dict, "positive_recall_attr_acc", "plots/positive_recall_attr_acc.png")
  # Negative labels
  bar_save(neg_acc_dict, "negative_recall_attr_acc", "plots/negative_recall_attr_acc.png")


def bar_save(dict_name, bar_name, save_path):

  # Extract the keys and values from the dictionary
  keys = list(dict_name.keys())
  values = list(dict_name.values())

  # Plot the bar plot
  plt.figure(figsize=(8, 6))
  plt.bar(keys, values)
  plt.title(f'Bar Plot of {bar_name} Values')
  plt.xlabel('attributes')
  plt.ylabel(f'{bar_name}')
  plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
  plt.grid(axis='y')
  plt.tight_layout()  # Adjust layout to ensure everything fits without overlap
  plt.savefig(save_path)




