import numpy as np
import pandas as pd
import scipy.sparse as sps
from sklearn.preprocessing import Normalizer


def get_preprocessed_dataframe(path="../data/", age_transformer=None, activity_transformer=None,
                               keep_warm_only=False):
    """
    Data preprocessing of URM and UCM.

    Default preprocessing step notes:
    - Region missing values are encoded with an additional class
    - Regions are one-hot encoded

    - Age missing values are imputed with the mean of the dataset + 1
    - Age is normalized with scikit-learn Normalizer()

    - User activity is transformed using log1p transformation
    - User activity, after log1p transformation, is normalized with scikit-learn Normalizer()

    :param path: path to the data folder
    :param age_transformer: transformer to be applied to age. Default is Normalizer(). It has to exploit
    same scikit learn interface.
    :param activity_transformer: transformer to be applied to the activity, after the log1p transformation.
    It has to export the same scikit learn interface
    :param keep_warm_only: keep only the warm users in the process
    :return: a pre-processed dataframe containing information regarding the region, the age, and
    the user activity profile length
    """
    # Reading the data and setting up the necessary things
    df_UCM_age = pd.read_csv(path + "data_UCM_age.csv")
    df_UCM_region = pd.read_csv(path + "data_UCM_region.csv")
    df_URM = pd.read_csv(path + "data_train.csv")
    user_list = df_URM['row']
    item_list = df_URM['col']
    URM_all = sps.coo_matrix((np.ones(len(df_URM)), (user_list, item_list)))
    URM_all = URM_all.tocsr()

    UCM_age = df_UCM_age[['row', 'col']]
    UCM_age.rename(columns={'row': 'user',
                            'col': 'age'},
                   inplace=True)
    UCM_region = df_UCM_region[['row', 'col']]
    UCM_region.rename(columns={'row': 'user', 'col': 'region'}, inplace=True)
    UCM_age_region = pd.merge(UCM_region, UCM_age, on="user", how="outer")

    # Missing values management
    UCM_age_region.loc[UCM_age_region['age'].isnull(), 'age'] = int(UCM_age_region['age'].mean()) + 1
    UCM_age_region.loc[UCM_age_region['region'].isnull(), 'region'] = -1

    # Adding user profile len information
    user_act = URM_all.sum(axis=1)
    total_users = np.arange(df_URM['row'].max() + 1)
    if keep_warm_only:
        warm_users = df_URM['row'].unique()
        warm_mask = np.in1d(total_users, warm_users)
        user_act_warm_users = user_act[warm_mask]
        df_user_act = pd.DataFrame(user_act_warm_users)
        df_user = pd.DataFrame(warm_users)
    else:
        user_act = np.array(user_act).squeeze()
        df_user_act = pd.DataFrame(user_act)
        df_user = pd.DataFrame(total_users)
    df_user.rename(columns={0: 'user'}, inplace=True)
    df_user_act.rename(columns={0: 'activity'},
                       inplace=True)
    df_user_act = pd.merge(df_user_act, df_user, left_index=True, right_index=True)
    merged = pd.merge(UCM_age_region, df_user_act, right_on="user", left_on="user")

    # Data pre-processing: One hot encoding region
    dfDummies = pd.get_dummies(merged[['user', 'region']]['region'], prefix='region')
    dfDummies = dfDummies.join(merged['user'])
    dfDummies = dfDummies.groupby(['user'], as_index=False).sum()
    merged = merged[['user', 'age', 'activity']]
    merged = pd.merge(merged, dfDummies, on="user")
    merged = merged.drop_duplicates(subset="user",
                                    keep='first', inplace=False)

    merged = merged.reset_index()

    # Data pre-processing: Normalizing age
    if age_transformer is None:
        age_transformer = Normalizer()
    temp = merged['age'].values.tolist()
    temp = [temp]
    age_transformer.fit(temp)
    res = age_transformer.transform(temp)
    age_norm = pd.DataFrame(res[0])
    age_norm.rename(columns={0: 'age_norm'},
                    inplace=True)
    age_norm = pd.merge(age_norm, df_user, left_index=True, right_index=True)
    merged = pd.merge(merged, age_norm, right_on="user", left_on="user")
    merged = merged[['user', 'activity', 'region_-1.0', 'region_0.0', 'region_2.0',
                     'region_3.0', 'region_4.0', 'region_5.0', 'region_6.0', 'region_7.0',
                     'age_norm']]

    # Data pre-processing: dealing with user activity highly-skewed distribution
    act = merged['activity'].values
    new_act = np.log1p(act)
    new_act = [new_act]
    if activity_transformer is None:
        activity_transformer = Normalizer()
    activity_transformer.fit(new_act)
    res = activity_transformer.transform(new_act)
    act_norm = pd.DataFrame(res[0])
    act_norm.rename(columns={0: 'act_norm'},
                    inplace=True)
    act_norm = pd.merge(act_norm, df_user, left_index=True, right_index=True)
    merged = pd.merge(merged, act_norm, right_on="user", left_on="user")
    merged.drop(columns=['activity'], inplace=True)

    # Re-setting index
    merged = merged.set_index('user')
    merged = merged.sort_index()

    return merged
