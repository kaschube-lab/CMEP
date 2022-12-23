"""
=============================================================================================================
Project: How much data do we need? Lower bounds of brain activation states to predict human cognitive ability
=============================================================================================================
Description
-----------
Code to split the data into different cross validation sets for training and testings

Author
------
Maren Wehrheim
marenwehrheim@gmail.com
Goethe University Frankfurt

License & Copyright
-------------------
Copyright 2022 Maren Wehrheim. All rights reserved.
This file is licensed to you under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License. You may obtain a copy
of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under
the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
OF ANY KIND, either express or implied. See the License for the specific language
governing permissions and limitations under the License.
"""

import numpy as np
from sklearn.model_selection import KFold


def cross_validation(y, sex, family_ID=None, split=10, rand_state=0):
    """
    Yields the train and test cross validation splits.
    :param y: behavioral data that should get splitted
    :param sex: Array containing the sex (1 if male, 2 if female)
    :param family_ID: array with unique family IDs
    :param split: The K-fold split that should be used
    :param rand_state: random state intitializer
    :yields: train and test indices
    """
    if type(family_ID) is not type(None):
        if split == len(y):
            print('leave one fam out')
            yield from leave_one_familiy_out_cv(family_ID)
        else:
            yield from cv_hcp(family_ID, splits=split, rand_state=rand_state)
    else:
        if split == len(y) or type(sex) is type(None):
            yield from cv(y, split)
        else:
            yield from strat_cv(y, sex, splits=split)


def cv(y, split):
    """
    Leave one out Cross Validation Loop
    :param y: behavioral data to be predicted
    :yield: train and test indices
    """
    cv = KFold(n_splits=split, shuffle=False)
    cv.get_n_splits(y)
    for train_indices, test_indices in cv.split(y):
        yield train_indices, test_indices


def strat_cv(y, sex, splits=10):
    """
    Creates a K-Fold stratified split into training and test data depending on iq and sex.
    :param y: behavioral data to be predicted
    :param sex: numpy array with geneder
    :param split: The K-fold split that should be used
    :yield: train and test indices
    """
    sorted_iq = np.argsort(y)
    women = np.where(sex == 2.0)[0]
    sorted_iq_women_index, sorted_iq_men_index = [], []
    for iq_index in sorted_iq:
        if iq_index in women:
            sorted_iq_women_index.append(iq_index)
        else:
            sorted_iq_men_index.append(iq_index)
    for i in range(splits):
        test_indices, train_indices = [], []
        women_test = sorted_iq_women_index[i::splits]
        train_indices.extend(list(set(sorted_iq_women_index) - set(women_test)))
        test_indices.extend(women_test)
        men_test = sorted_iq_men_index[i::splits]
        train_indices.extend(list(set(sorted_iq_men_index) - set(men_test)))
        test_indices.extend(men_test)
        yield train_indices, test_indices


def leave_one_familiy_out_cv(family_ID):
    """
    Cross validation for all families, i.e. always removing a complete family as test data.
    :param family_ID: array with unique family IDs
    :yield: train and test indices
    """
    unique_IDs = np.unique(family_ID)
    for unique_fam in unique_IDs:
        train_indices = np.where(family_ID != unique_fam)[0]
        test_indices = np.where(family_ID == unique_fam)[0]
        yield train_indices, test_indices


def cv_hcp(family_ID, splits=10, rand_state=0):
    """
    Cross validation for all families, i.e. always removing a complete family as test data.
    :param family_ID: array with unique family IDs
    :param split: the K-fold split that should be used
    :param rand_state: random state intitializer
    :yield: train and test indices
    """
    unique_IDs = np.unique(family_ID)
    cv = KFold(n_splits=splits, shuffle=True, random_state=rand_state)
    cv.get_n_splits(unique_IDs)
    for train_index, test_index in cv.split(unique_IDs):
        train_unique_IDs, test_unique_IDs = unique_IDs[train_index], unique_IDs[test_index]
        train_indices = np.where(np.isin(family_ID, train_unique_IDs) == True)[0]
        test_indices = np.where(np.isin(family_ID, test_unique_IDs) == True)[0]
        yield train_indices, test_indices
