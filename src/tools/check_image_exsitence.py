import os
import pandas as pd
import numpy as np

def check_file_existance(dataframe_path, image_source_path):
    image_dataframe_ids = list(pd.read_excel(dataframe_path)["ID"])
    image_dir_names = list(os.listdir(image_source_path))

    result = [image_dataframe_ids[i] for i in range(len(image_dataframe_ids)) \
              if f"{image_dataframe_ids[i]}.jpg" not in image_dir_names]

    print(result)


if __name__ == "__main__":
    check_file_existance("data/NATIVE/Native_dataset.xlsx", "data/NATIVE/IDs")