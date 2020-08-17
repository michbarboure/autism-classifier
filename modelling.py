from sklearn.feature_selection import SelectKBest, chi2, f_classif, SelectFdr, VarianceThreshold
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from split import stratify_by_subject
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from accuracy import subject_level_accuracy_score
import numpy as np
from selected_features import select_feat
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegressionCV
import os


def import_merged_data(feather_file="BD_subset_06_balanced.feather"):

    print('Reading data...')
    data = pd.read_feather(feather_file)
    print(data.shape)

    return data


def add_pca_feats(data):

    _data = data.copy()

    # drop the `id` col if it's there
    if 'id' in data.columns:
        data.drop('id', axis=1, inplace=True)

    # list of things not supposed to be in X
    bads = ['is_OD', 'B_ID', 'event', 'birth_year', 'n_initial_epochs',
            'is_male', 'is_asd', 'EXCLUDE', 'age', 'age_bin', 'clean_ratio']

    TARGET_COL = 'is_asd'

    X = data.drop(bads, axis=1)
    y = data[TARGET_COL]

    print('Generating 100 features using PCA ...')

    pca = PCA(n_components=100)
    scaler = StandardScaler()
    _X = scaler.fit_transform(X)
    _X_proj = pca.fit_transform(_X)
    _X = pd.DataFrame(data=_X_proj)
    _X_merge = pd.concat([_data, _X], axis=1)

    # Make all column names into strings
    _X_merge.columns = [str(c) for c in _X_merge.columns]

    return _X_merge


def feature_selection_rfe(data, group_cols=['B_ID'], feat_count=1, verbose=0):

    # drop the `id` col if it's there
    if 'id' in data.columns:
        data.drop('id', axis=1, inplace=True)

    # list of things not supposed to be in X
    bads = ['is_OD', 'B_ID', 'event', 'birth_year', 'n_initial_epochs',
            'is_male', 'is_asd', 'EXCLUDE', 'age', 'age_bin', 'clean_ratio']

    TARGET_COL = 'is_asd'
    
    _data = data.groupby(group_cols).mean().reset_index()
    X = _data.drop(bads, axis=1)
    y = _data[TARGET_COL]
    print(X.shape)

    print('Feature selection using Recursive Feature Elimination...')
    from sklearn.feature_selection import RFE
    from sklearn.tree import DecisionTreeClassifier

    model = DecisionTreeClassifier()
    rfe = RFE(estimator=model, n_features_to_select=feat_count,
              step=1, verbose=verbose).fit(X, y)
    feats = pd.DataFrame.from_dict({X.columns[i]: v for i, v in enumerate(
        rfe.ranking_)}, orient='index', columns=['ranking'])
    feats.sort_values(by='ranking', inplace=True)

    print(feats.head(50))

    feats.to_csv(f"rfe_feature_ranking_{'-'.join(group_cols)}.csv", index=True)

    return list(feats.index)


def model(data):

    import warnings
    from sklearn.exceptions import UndefinedMetricWarning
    warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)

    algorithms = {'RF': RandomForestClassifier, 'XG': XGBClassifier}
    results = {'algorithm': [], 'n_features': [], 'group_cols': [], 'rfe_group_cols': [], 
    'agg_class': [], 'accuracy': [], 'recall': [], 'precision': [], 'true_neg': [], 
    'false_pos': [], 'false_neg': [], 'true_pos': []}

    bads = ['is_OD', 'B_ID', 'event', 'birth_year', 'n_initial_epochs',
            'is_male', 'is_asd', 'EXCLUDE', 'age', 'age_bin', 'clean_ratio']

    for rfe_group_cols in [['B_ID'], ['B_ID', 'event'], ['B_ID', 'is_OD']]:
        print(f"\n\nPerforming RFE on {rfe_group_cols}...\n\n")
        # Get the names of the 1000 best non-bads features (ranked)
        ranked_feats = feature_selection_rfe(data, group_cols=rfe_group_cols, verbose=0)

        for i in tqdm(range(5), desc='Resamples'):  # Monte carlo resampling
            # train-test split (stratified)

            for feat_count in tqdm([20, 30, 40, 50, 60, 80, 100, 150, 200, 250, 300], desc='n_features'):
                print('Feature count:', feat_count)

                best_cols = ranked_feats[:feat_count]

                _X_merge = data[[*best_cols, *bads]]
                print(_X_merge.shape)

                # Add event-describing features
                _X_merge['is_click_event'] = _X_merge['event'].apply(
                    lambda x: 1 if x in [16, 17] else 0)

                train, test = stratify_by_subject(
                    _X_merge, test_percent=0.4, seed=None)

                for group_cols in [['B_ID', 'is_OD'], ['B_ID', 'event'], ['B_ID']]:
                    print('Grouping by:', group_cols)

                    for agg_class in [1, 3]:
                        _X_train, _X_test, _y_train, _y_test = prepare_aggregation_class(
                            train, test, agg_class, bads, group_cols=group_cols)

                        for alg_name, alg in algorithms.items():
                            print(_X_train.shape, _y_train.shape, _X_test.shape)
                            y_pred = alg().fit(_X_train, _y_train).predict(_X_test)

                            results['algorithm'].append(alg_name)
                            results['n_features'].append(feat_count)
                            results['agg_class'].append(agg_class)
                            results['group_cols'].append(str(group_cols))
                            results['rfe_group_cols'].append(str(rfe_group_cols))

                            accuracy, recall, precision, tn, fp, fn, tp = evaluate_for_agg_class(
                                test, y_pred, agg_class)

                            results['accuracy'].append(accuracy)
                            results['recall'].append(recall)
                            results['precision'].append(precision)
                            results['true_neg'].append(tn)
                            results['false_pos'].append(fp)
                            results['false_neg'].append(fn)
                            results['true_pos'].append(tp)

                checkpoint = pd.DataFrame(results)
                best = checkpoint[checkpoint['accuracy']
                                  == checkpoint['accuracy'].max()]
                print(best)
                CHECK_DIR = 'results_checkpoints'
                if not os.path.exists(CHECK_DIR):
                    os.makedirs(CHECK_DIR)
                checkpoint.to_csv(CHECK_DIR+'/latest.csv', index=False)

    df = pd.DataFrame(results)
    return df


def prepare_aggregation_class(train, test, agg_class, bads, group_cols=['B_ID', 'event'], target_col='is_asd'):
    # aggregate train and/or test data according to agg_class
    # NB: Don't modify original

    # added 'is_od' for the subset with both tasks
    from copy import deepcopy
    _bads = deepcopy(bads)
    _bads.remove('is_OD')

    if agg_class == 0:
        _X_train = train.groupby(group_cols).mean().reset_index()
        _X_test = test.groupby(group_cols).mean().reset_index()

    if agg_class == 1:
        _X_train = train.groupby(group_cols).mean().reset_index()
        _X_test = test.copy()

    if agg_class == 2:
        _X_train = train.copy()
        _X_test = test.groupby(group_cols).mean().reset_index()

    if agg_class == 3:
        _X_train = train.copy()
        _X_test = test.copy()

    _y_train = _X_train[target_col]
    _y_test = _X_test[target_col]

    _X_train = _X_train.drop(_bads, axis=1)
    _X_test = _X_test.drop(_bads, axis=1)

    # Organise columns in X_train and X_test alphabetically to prevent misalignment
    _X_train = _X_train.sort_index(axis=1)
    _X_test = _X_test.sort_index(axis=1)

    # for c1, c2 in zip(_X_train.columns, _X_test.columns):
    #     print(c1, c2)

    return _X_train, _X_test, _y_train, _y_test


def evaluate_for_agg_class(test, y_pred, agg_class):
    # returns acc, sens, spec using appropriate calcs
    # Depends on which aggregation class you've set up

    # labels for subject_level_accuracy_score
    labels = pd.read_csv("conf/labels_updated.csv")

    t = 0.5
    y_true_strat, y_pred_strat = subject_level_accuracy_score(
        y_pred, test, labels, thresh=t)

    accuracy = accuracy_score(y_true_strat, y_pred_strat)
    recall = recall_score(y_true_strat, y_pred_strat)
    precision = precision_score(y_true_strat, y_pred_strat)
    tn, fp, fn, tp = confusion_matrix(y_true_strat, y_pred_strat).ravel()

    return accuracy, recall, precision, tn, fp, fn, tp

