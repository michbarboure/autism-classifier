import pandas as pd
from sklearn.metrics import accuracy_score


def subject_level_accuracy_score(y_pred, test, labels, thresh=0.5):
    y_pred_df = pd.DataFrame(data=y_pred)
    y_pred_df.rename({0: 'y_pred'}, axis='columns', inplace=True)
    test_id = test['B_ID'].reset_index(drop=True)
    merge = pd.concat([y_pred_df, test_id], axis=1)
    pred = merge.groupby(['B_ID']).mean()
    pred['consensus'] = pred['y_pred'].apply(lambda p: 1 if p >= thresh else 0)
    score_df = pred.merge(labels[['is_asd', 'B_ID']], how='left', on='B_ID')
    
    return score_df['is_asd'].values, score_df['consensus'].values
