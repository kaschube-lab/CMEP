"""
=============================================================================================================
Project: How much data do we need? Lower bounds of brain activation states to predict human cognitive ability
=============================================================================================================
Description
-----------
Covariance maximizing eigenvector-based prediciton (CMEP) framework.

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

from crossValidation import cross_validation
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


@ignore_warnings(category=ConvergenceWarning)
def predict_cmep(X, y, split=10, sex=None, family_ID=None, rand_state=0):
    """
    Predicts intelligence from sFC
    :param X (np array): data matrix. Here functional connectivity matrix for each participant, shape: n_participants, n_nodes, n_nodes
    :param y (np array): 1D numpy array containing variable of interest. Here IQ values for each participant
    :param split (int): Number of cross validation splits (default 10)
    :param sex (np array): 1D-array containing the sex values for each participant (1=men, 2=women) used for stratified cross validation.
    :return: predicted values
    """
    # initialize all of the variables and arrays
    n_participants, n_nodes = X.shape[:2]
    y_pred = np.zeros(n_participants)

    alphas = [0.01, 0.02, 0.05, 0.1, 0.5, 1]
    l1_ratios = [0.01, 0.05, 0.1, 1 / 3, 0.5, 0.7, 0.9, 0.95]

    
    # Prediction CV Loops
    for (train_indices, test_indices) in cross_validation(y, sex=sex, family_ID=family_ID, 
                                                          split=split, rand_state=rand_state):
        X_train, X_test, y_train = X[train_indices], X[test_indices], y[train_indices]
        
        # Calculate Eigenvectors of Q and the variances to each connectivity matrix using those
        train_data, test_data = create_features(X_train, X_test, y_train, n_nodes)

        # Predict & Set best parameters in a 5-fold CV Loop
        en = ElasticNetCV(cv=5, random_state=0, alphas=alphas, l1_ratio=l1_ratios, max_iter=10000)
        en.fit(train_data, y_train)
        pred_fold = en.predict(test_data)
        y_pred[test_indices] = pred_fold
    return y_pred


def calc_M(X, y):
    """
    Calculate the y-weighted matrix M (Eq. 7)
    :param X (np array): data matrix. Here sFC input
    :param y (np array): intelligence test scores
    :return: weighted matrix M
    """
    y_mean_adj = y - np.mean(y)
    X_weighted = np.zeros_like(X)
    nb_samples = len(X)
    for i in range(nb_samples):
        X_weighted[i] = np.multiply(y_mean_adj[i], X[i])
    M = np.mean(X_weighted, axis=0)
    return M


def sort_evecs(evecs, evals):
    """
    Sort the eigenvectors given their eigenvalue
    :param evecs (np array): square eigenvector matrix with vectors in the rows of matrix
    :param evals (np array): full eigenvalue vector
    :return: sorted eigenvector matrix and eigenvalue array
    """
    indices_sorted = np.argsort(np.abs(evals))[::-1]
    return evecs[indices_sorted], evals[indices_sorted]


def calc_eigvec_decomposition(M):
    """
    Decomposes the input matrix M into its eigenvectors and eigenvalues
    :param M (np array): average y-weighted matrix
    :return: eigenvectors & -values sorted given their eigenvalues
    """
    evals, evecs = np.linalg.eig(M)
    evecs = evecs.T  # new shape: (n_evecs, n_entries)
    sorted_evecs, sorted_evals = sort_evecs(evecs, evals)
    return sorted_evecs, sorted_evals


def calc_features(X, evecs, n_features):
    """
    Calculates the variance features for each of the input samples
    :param X (np array): data matrix. Here functional connectivity matrix for each participant, shape: n_participants, n_nodes, n_nodes
    :param evecs (np array): eigenvectors
    :param n_features (int): number of features that should get created, i.e., number of eigenvectors that should be used
    :return: feature matrix for all participants
    """
    features = np.zeros((len(X), n_features))
    for i in range(len(X)):
        feats = []
        for j in range(n_features):
            feat = evecs[j].T.dot(X[i]).dot(evecs[j])
            feats.append(feat)
        features[i] = feats
    return features


def create_features(X_train, X_test, y_train, n_features):
    """
    Generates the features of the input data
    :param X_train (np array): data matrix. Here functional connectivity matrix for each participant in train set
    :param X_test (np array): data matrix. Here functional connectivity matrix for each participant in test set
    :param y_train (np array): variable of interest in train set
    :param n_features (int): Number of features that should be created
    :return: training at test feature matrices
    """
    M = calc_M(X_train, y_train)
    sorted_evecs, _ = calc_eigvec_decomposition(M)
    features_train = calc_features(X_train, sorted_evecs, n_features)
    features_test = calc_features(X_test, sorted_evecs, n_features)
    return features_train, features_test

