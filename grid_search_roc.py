import ipdb
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, mean_absolute_error
from split import stratify_by_subject
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.svm import SVC, SVR
from tqdm import tqdm
import numpy as np
import pandas as pd


def find_regression_thresh(y_train, y_train_cla, y_pred_reg):
    """ Finds best threshold + corresponding predictions for regression-classification conversion.
    """
    accuracies = []
    for thresh in range(0, 113):
        pred = [1 if p > thresh else 0 for p in y_train]
        accuracies.append(accuracy_score(y_train_cla, pred))

    thresh_param = np.argmax(accuracies)  # the thresh with best accuracy
    y_pred = [1 if p > thresh_param else 0 for p in y_pred_reg]

    return thresh_param, y_pred

# Statics
BADS = ['MR_AQ28_Tot', 'B_ID', 'birth_year', 'n_initial_epochs',
        'is_male', 'is_asd', 'EXCLUDE', 'age', 'age_bin', 'clean_ratio']
DATASET = 'Balanced_360initepoch_both_19-07-2020.csv'
OUTFILE = 'roc_results_balanced_pre_scaled_03-08-2020.csv'
TEST_SPLIT_SIZE = 0.3
USE_CACHED_RFE = True

# Read the dataset
data = pd.read_csv(DATASET)
print(data.shape)

targets = {'Classification': 'is_asd',
           'Regression': 'MR_AQ28_Tot'}

rfe_algs = {'Regression': DecisionTreeRegressor,
            'Classification': DecisionTreeClassifier}

algorithms = {
    'Regression': {
        'Lasso_reg': Lasso,
        'SVM_reg': SVR,
        'DT_reg': DecisionTreeRegressor,
        'RF_reg': RandomForestRegressor,
        'XG_reg': XGBRegressor
    },
    'Classification': {
        'LogReg_cla': LogisticRegression,
        'SVM_cla': SVC,
        'DT_cla': DecisionTreeClassifier,
        'RF_cla': RandomForestClassifier,
        'XG_cla': XGBClassifier
    }
}

alg_hypers = {
    'Lasso_reg': {},
    'SVM_reg': {},
    'DT_reg': {},
    'RF_reg': {'n_estimators': 100},
    'XG_reg': {},
    'LogReg_cla': {'penalty': 'l1', 'solver': 'liblinear'},
    'SVM_cla': {},
    'DT_cla': {},
    'RF_cla': {'n_estimators': 100},
    'XG_cla': {},
}

optimal_n_feats = {
    'Lasso_reg': 9,
    'SVM_reg': 2,
    'DT_reg': 12,
    'RF_reg': 7,
    'XG_reg': 5,
}


results = {'algorithm': [], 'learn_type': [], 'is_OD': [], 'n_features': [], 'threshold': [],
           'mae': [], 'accuracy': [], 'recall': [], 'precision': [], 'true_neg': [],
           'false_pos': [], 'false_neg': [], 'true_pos': []}

# Scale the features (to help SVR)
sc = StandardScaler()
feature_col_names = list(data.drop([*BADS, 'is_OD'], axis=1).columns)
data[feature_col_names] = sc.fit_transform(data[feature_col_names])

# RFE: Get the names of the 1000 best non-bads features (ranked)
ranked_feats = {'Regression': {0: None, 1: None}, 'Classification': {0: None, 1: None}}
for learn_type in ['Regression', 'Classification']:
    rfe_alg = rfe_algs[learn_type]
    for od_task in [1, 0]:
        if not USE_CACHED_RFE:
            print(f"RFE for {learn_type}, is_OD = {od_task}")
            _data = data[data.is_OD == od_task]
            X, y = _data.drop(BADS, axis=1), _data[targets[learn_type]].values.ravel()
            rfe = RFE(estimator=rfe_alg(), n_features_to_select=1, step=1).fit(X, y)
            feats = pd.DataFrame.from_dict({X.columns[i]: v for i, v in enumerate(rfe.ranking_)}, orient='index', columns=['ranking'])
            feats.sort_values(by='ranking', inplace=True)
            feats.to_csv(f"rfe_feature_ranking_OD{od_task}_{learn_type}.csv", index=True)
        else:
            feats = pd.read_csv(f"rfe_feature_ranking_OD{od_task}_{learn_type}.csv")
            feats.set_index('Unnamed: 0', inplace=True)

        ranked_feats[learn_type][od_task] = list(feats.index)

# Perform grid search
for od_task in [0]:
    if od_task == 1:
        print("\nOD TASK\n\n")
    else:
        print("\nBD TASK\n\n")

    for i in tqdm(range(100), desc='Resamples'):  # Monte carlo resampling

        for learn_type in ['Regression']:

            # Set appropriate variables
            algs = algorithms[learn_type]
            target = targets[learn_type]

            _data = data[data.is_OD == od_task]

            # Stratified train-test split
            train, test = stratify_by_subject(
                _data, test_percent=TEST_SPLIT_SIZE, seed=None)
            X_train, X_test = train.drop(BADS, axis=1), test.drop(BADS, axis=1)
            train_bids, test_bids = train['B_ID'], test['B_ID']
            y_train, y_test_cla, y_test_reg = train[target], test['is_asd'], test['MR_AQ28_Tot']
            y_train_cla = train['is_asd']

            for alg_name, alg in algs.items():
                
                feat_count = optimal_n_feats[alg_name]
                
                best_cols = ranked_feats[learn_type][od_task][:feat_count]

                _X_train = X_train[['is_OD', *best_cols]]
                _X_test = X_test[['is_OD', *best_cols]]

                hypers = alg_hypers[alg_name]
                _alg = alg(**hypers)
                y_pred = _alg.fit(_X_train, y_train).predict(_X_test)

                # Initialise variables to non-meaningful values
                mae = -1.0
                asd_truths = y_test_cla

                # Convert regression to classification
                if learn_type == 'Regression':

                    preds = pd.DataFrame(
                        {'B_ID': test_bids, 'pred': y_pred, 'is_asd': y_test_cla, 'MR_AQ28_Tot': y_test_reg})
                    preds = preds.groupby('B_ID').mean()

                    y_pred_reg = preds['pred']
                    mae = mean_absolute_error(preds['MR_AQ28_Tot'], y_pred_reg)
                    asd_truths = preds['is_asd']

                    # Building ROC data by varying the threshold
                    for thresh in range(0, 113):
                        y_pred = [1 if p >= thresh else 0 for p in y_pred_reg]

                        accuracy = accuracy_score(asd_truths, y_pred)
                        precision = precision_score(asd_truths, y_pred)
                        recall = recall_score(asd_truths, y_pred)
                        tn, fp, fn, tp = confusion_matrix(
                            asd_truths, y_pred).ravel()

                        results['learn_type'].append(learn_type)
                        results['is_OD'].append(od_task)
                        results['algorithm'].append(alg_name)
                        results['n_features'].append(feat_count)
                        results['threshold'].append(thresh)
                        results['accuracy'].append(accuracy)
                        results['mae'].append(mae)
                        results['precision'].append(precision)
                        results['recall'].append(recall)
                        results['true_neg'].append(tn)
                        results['false_pos'].append(fp)
                        results['false_neg'].append(fn)
                        results['true_pos'].append(tp)

                checkpoint = pd.DataFrame(results)
                checkpoint.to_csv('latest.csv', index=False)

df = pd.DataFrame(results)
df.to_csv(OUTFILE, index=False)
ipdb.set_trace()
