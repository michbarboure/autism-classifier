import pandas as pd
import random
from pathlib import Path


def _attach_int_ids(df):
    add_columns = df['B_ID'].str.split('_', expand=True)
    df['id_as_int'] = add_columns[1].values
    df['id_as_int'] = df['id_as_int'].astype(dtype='int')

    all_ids = df['id_as_int'].unique().tolist()

    return df, all_ids


def _split(all_ids):
    even_list = []
    odd_list = []

    for i in all_ids:
        if (i % 2 == 0):
            even_list.append(i)
        else:
            odd_list.append(i)

    # print("Even lists:", even_list)
    # print("Odd lists:", odd_list)
    return even_list, odd_list


def _partition(list_to_divide, test_percent=0.3, seed=42):

    perc_portion = int(round(test_percent*len(list_to_divide)))
    shuffled = list_to_divide[:]

    if seed is not None:
        random.seed(seed)
    random.shuffle(shuffled)

    return shuffled[perc_portion:], shuffled[:perc_portion]


def stratify_by_subject(df, test_percent=0.3, seed=42):
    """Split epochs on a per-subject basis instead of a per-row basis
    to prevent data leakage.

    Parameters
    ----------
    df : [type]
        Combined dataset with engineered features.
    test_percent : float, optional
        The fraction of the split for testing, by default 0.3
    seed : int, optional
        Random seed for splitting, by default 42

    Returns
    -------
    (train, test) as dataframes
    """
    _df = df.copy()
    _df, all_ids = _attach_int_ids(_df)
    evens, odds = _split(all_ids)
    trains, tests = [], []
    for id_list in [evens, odds]:
        train_ids, test_ids = _partition(
            id_list, test_percent=test_percent, seed=seed)
        trains.extend(train_ids)
        tests.extend(test_ids)
    # print(len(trains), len(tests))

    train = _df[_df['id_as_int'].isin(trains)]
    test = _df[_df['id_as_int'].isin(tests)]

    train = train.drop('id_as_int', axis=1)
    test = test.drop('id_as_int', axis=1)

    # Shuffle the rows in each dataframe
    train = train.sample(frac=1)
    test = test.sample(frac=1)

    return train, test
