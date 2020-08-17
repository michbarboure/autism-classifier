import sys
import glob
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import pyarrow.feather as feather

def stitch(src_path, dest_path):
    """Read csv files and concatenate them along the row-axis (axis=0); the
    dest path needs to be the name of the file (e.g. "merge.csv"); feather files
    do not need an extension
    """

    # src_path = Path(src_path)
    src_path, dest_path = Path(src_path), Path(dest_path)

    file_list = list(src_path.glob('*.csv'))

    #TODO remove lim
    # file_list = file_list[:3]

    df_list = []
    for file_path in tqdm(file_list):
        dataframe = pd.read_csv(file_path)
        df_list.append(dataframe)
    print(len(df_list))

    df = pd.concat(df_list, axis=0, ignore_index=True)

    # df.to_csv(dest_path, index=False)

    df.to_feather(dest_path)