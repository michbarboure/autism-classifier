import ipdb
import pandas as pd
import numpy as np
from tqdm import tqdm


def reading_data():

    print('Reading data...')
    data = pd.read_feather("BD_merge_15.feather")
    print(data.shape)

    print('Attaching labels and metadata...')
    # combined metadata
    metadata = pd.read_csv("metadata/metadata_all_updated.csv")
    print(metadata.shape)

    # labels
    labels = pd.read_csv("conf/labels_updated.csv")
    print(labels.shape)

    # join all of the above 3
    data = data.merge(labels, how='left', on='B_ID')
    print(data.head())
    print(data.shape)

    meta_subset = ['B_ID', 'is_OD', 'clean_ratio', 'n_initial_epochs']
    data = data.merge(metadata[meta_subset], how='left', on=['B_ID', 'is_OD'])
    print(data.head())
    print(data.shape)

    INITIAL_EPOCH_CUTOFF = 360
    CLEAN_RATIO = 0.6
    import ipdb; ipdb.set_trace()
    print(f'Drop subjects with a clean ratio below {CLEAN_RATIO}, and number of initial epochs below {INITIAL_EPOCH_CUTOFF}')
    data = data[(data.clean_ratio >= CLEAN_RATIO) & (data.n_initial_epochs >= INITIAL_EPOCH_CUTOFF)]
    data.reset_index(drop=True, inplace=True)
    print(data.shape)

    return data

# BD balancing downsampling
# exceptions = ['B_005', 'B_019', 'B_067', 'B_069', 'B_113', 'B_127', 'B_141', 
# 'B_157', 'B_163', 'B_169', 'B_171', 'B_191']

# OD balancing downsampling
# exceptions = ['B_005', 'B_006', 'B_009', 'B_019', 'B_029', 'B_069', 'B_113', 'B_127', 'B_141', 'B_169', 'B_191']

# bal = data[~data['B_ID'].isin(exceptions)]
# add = data_05[data_05['B_ID'].isin('B_022')]
# merge = pd.concat([bal, add], axis=0)

# asd = final.loc[final['is_asd'] == 1] 
# non = final.loc[final['is_asd'] == 0]

# Regression dataset ASD downsampling
df = pd.read_csv("BOTH_no_selection_feat_averaged_prior.csv")
spq = pd.read_csv("SPQdata.csv")
merge = df.merge(spq[['MR_AQ28_Tot', 'B_ID']], how='left', on='B_ID')
data = merge.dropna(subset=['MR_AQ28_Tot'])
INITIAL_EPOCH_CUTOFF = 360
data_2 = data[(data.n_initial_epochs >= INITIAL_EPOCH_CUTOFF)]
data_2.drop('id', axis=1, inplace=True)
data_3 = data_2.drop_duplicates(subset='B_ID')
exceptions_1 = ['B_049', 'B_193', 'B_157', 'B_095', 'B_171', 'B_125', 'B_139']
bal = data_3[~data_3['B_ID'].isin(exceptions_1)]
exceptions_2 = ['B_079', 'B_017', 'B_021', 'B_029', 'B_033', 'B_037', 'B_043', 'B_055', 'B_059', 'B_073', 'B_089']
exceptions_3 = ['B_005', 'B_183', 'B_013', 'B_141', 'B_035', 'B_083', 'B_061', 'B_067', 'B_085', 'B_091', 'B_097', 'B_101', 'B_103', 'B_113', 'B_127', 'B_153']
bal_2 = bal[~bal['B_ID'].isin(exceptions_2)]
bal_3 = bal_2[~bal_2['B_ID'].isin(exceptions_3)]
bal_3.is_asd.value_counts()
asd = bal_3.loc[bal_3['is_asd'] == 1]
non = bal_3.loc[bal_3['is_asd'] == 0]
asd.is_male.value_counts()
non.is_male.value_counts()
bal_list = bal_3.B_ID.tolist()

final = data_2[data_2['B_ID'].isin(bal_list)]
final.to_csv("reg_test_subset_balanced_19-07-2020.csv", index=False) 


