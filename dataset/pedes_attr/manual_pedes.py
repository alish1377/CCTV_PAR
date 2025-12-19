import glob
import os
import pickle

import numpy as np
import torch.utils.data as data
from PIL import Image

import numpy
import pandas as pd
import time

import torch

class ManualPedes(data.Dataset):
    def __init__(self, dataframe, split, root_path, transform, attrs):
        self.dataframe = dataframe.loc[split]
        self.root_path = root_path
        self.transform = transform
        self.attrs = attrs
        self.label = None
        self.attr_num = len(attrs)
        self.eval_attr_num = self.attr_num

    def __len__(self):
        return (len(self.dataframe))

    def __getitem__(self, index):
        img_path = os.path.join(self.root_path, f"{self.dataframe.iloc[index, 0]}.jpg")
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)

        # label = torch.tensor(self.dataframe.loc[self.dataframe.index[self.dataframe['ID'] \
        #                             == self.dataframe.iloc[index, 0]].tolist()[0], self.attrs])


        label = torch.tensor(self.dataframe.iloc[
                                 (self.dataframe['ID'] == self.dataframe.iloc[index, 0]).to_numpy().nonzero()[0][0],
                                 self.dataframe.columns.get_indexer(self.attrs)
                             ])

        label = label.to(torch.float32)

        #print(self.dataframe.iloc[index, 0])
        return img, label, ""
