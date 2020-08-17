import mne
import sys
import glob
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from datetime import datetime
from tsfresh.feature_extraction import extract_features, ComprehensiveFCParameters, MinimalFCParameters
from sklearn.preprocessing import OneHotEncoder


def load_fif(file_src):
    """Load epoch .fif file and save path filename as Path object
    """
    file_name = Path(file_src)
    epochs = mne.read_epochs(file_name)

    return epochs, file_name


def fif_convert_df(epochs):
    """Convert epoch .fif file as a dataframe and reset index
    """
    df = epochs.to_data_frame(
        picks=None,
        index=None,
        long_format=False,
    )

    df.reset_index(inplace=True)

    return df


def df_modify(df, target_file):
    """Narrow down channels of interest, create 'ID_unique' column that is
    task-subject-condition-epoch specific, and drop unwanted columns before
    doing feature extraction

    e.g.
    task: OD
    subject: 079
    condition: 16
    epoch: 103
    """

    # channels = ['A27', 'A25', 'B32', 'B30', 'B15', 'A32', 'B12', 'A9']
    # channels_new = ['A28', 'A29', 'A30', 'B31', 'A26']

    channels_final_9 = ['A25', 'A26', 'A27', 'A28', 'A29', 'A30', 'B30', 'B31', 'B32']

    _df = df.copy()
    _df = _df.groupby('time').mean().reset_index()
    _df['task'] = target_file.name.split('_')[0]
    _df['subject'] = target_file.name.split('_')[1]
    _df['ID_unique'] = _df['task'].map(str) + '_' + _df['subject'].map(str)
    drop = ['EXG1', 'EXG2', 'EXG3', 'EXG4',
            'Status', 'subject', 'epoch']
    _df.drop(columns=drop, inplace=True)

    _df = _df[['ID_unique', 'time', *channels_final_9]]

    return _df


def df_modify_after_extraction(df):
    """After feature extraction, the dataframe is modified further: "ID_unique"
    column (renamed by tsfresh to 'id') is split into 'is_OD' (binary), 'B_ID'
    (subject ID), and 'event' (e.g. 16) columns; 'event' column is changed to
    integer format
    """
    _df = df.copy()
    add_columns = _df.reset_index()['id'].str.split('_', expand=True)
    _df['is_OD'] = add_columns[0].apply(lambda p: 1 if p == 'OD' else 0).values
    _df['B_ID'] = 'B_' + add_columns[1].values
    # _df['event'] = add_columns[2].values
    # _df['event'] = _df['event'].astype(dtype='int')

    return _df


def df_convert_sample(file_src):
    """One function that brings all the above functions together (except
    df_modify_after_extraction)
    """
    target_file = Path(file_src)

    epochs, file_name = load_fif(target_file)
    df = fif_convert_df(epochs)
    df_mod = df_modify(df, file_name)

    return df_mod


def df_convert_batch(src_path, dest_path):
    """Batch function for df_convert_sample; all .fif files are initially found
    and added to a list which is then read and each .fif file is processed in 
    succession.

    tsfresh is called to perform feature extraction and df is modified a final
    time before being saved as a csv with a unique timestamp.
    """
    src_path, dest_path = Path(src_path), Path(dest_path)

    fif_files = []
    for fif in src_path.glob('**/*.fif'):
        fif_files.append(fif)
    # print(fif_files)

    dfs = []

    for file_path in tqdm(fif_files, desc="FIF converting"):

        tstamp = datetime.now().strftime("%y-%m-%d-%H-%M")
        sid = file_path.name.split('.')[0]
        df = df_convert_sample(file_path)

        df_X = extract_features(
            df,
            column_id='ID_unique',
            column_sort='time',
            default_fc_parameters=ComprehensiveFCParameters(),
            # default_fc_parameters=MinimalFCParameters()
        )
        print(df.shape, df_X.shape)

        df_X_mod = df_modify_after_extraction(df_X)

        # df_X_mod.to_csv(dest_path / f'{sid}_{tstamp}.csv')
        # Merge now instead of saving each
        dfs.append(df_X_mod)
    df_merged = pd.concat(dfs)
    df_merged.to_csv(dest_path / f'merged_{tstamp}.csv') 


if __name__ == '__main__':
    """System arguments should be source path (1) and destination path (2)
    """
    if len(sys.argv) >= 2:
        target_file = Path(sys.argv[1])
    else:
        raise ValueError("You need to specify paths (and optional limits)")

    if len(sys.argv) == 3:
        df_convert_batch(sys.argv[1], sys.argv[2])
