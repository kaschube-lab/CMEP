"""
============================================================================
Project: Predicting Intelligence from Static and Dynamic Brain Connectivity
============================================================================
Description
-----------
Utility functions

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
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression

np.random.seed(31415)


def create_FC(raw_signal):
    """
    Create a functional connectivity matrix for each participant from the raw bold signal.
    :param raw_signal (np array): raw bold signal for each participant. Shape: n_participants x n_nodes x T
    :return: functional connectivity matrices. Shape: n_participants x n_nodes x n_nodes
    """
    n_participants, n_nodes, _ = raw_signal.shape
    FC = np.zeros((n_participants, n_nodes, n_nodes))
    for i, raw_sample in enumerate(raw_signal):
        FC[i] = np.corrcoef(raw_sample)
    return FC


def fisher_z_transform(FC):
    """
    Fisher z-transform the input matrix
    :param FC: connectivity matrices of all participants
    :return: fisher z-transformed matrix
    """
    fisher_z_FC = np.zeros(FC.shape)
    for n in range(len(FC)):
        for row in range(1, FC.shape[1]):
            for col in range(row + 1):
                if row != col:
                    z = np.arctanh(FC[n, row, col])
                    # Catch too high values
                    if z > 20:
                        z = 20
                    elif z < -20:
                        z = -20
                    fisher_z_FC[n, row, col] = z
                    fisher_z_FC[n, col, row] = z
    fisher_z_FC = np.nan_to_num(fisher_z_FC)
    return fisher_z_FC
    

def control_FC(FC, control):
    """
    Controls FC for confounding effects
    :param FC: data matrix for each participant. Here functional connectivity
    :param control: array containing the variable to cotnrol for e.g., movement
    :return: controlled FC
    """
    print("Control connectivity")
    n_nodes = FC.shape[1]
    FC_regressed = FC.copy()
    for row in range(n_nodes):
        for col in range(row):
            if col != row:
                y_edge = FC_regressed[:, col, row].reshape((-1, 1))
                if control.ndim == 1:
                    control = control.reshape((-1, 1))
                reg = LinearRegression().fit(control, y_edge)
                for i in range(len(reg.coef_[0])):
                    y_edge -= reg.coef_[0, i] * control[..., i][..., np.newaxis]
                FC_regressed[:, row, col] = y_edge[:, 0]
                FC_regressed[:, col, row] = y_edge[:, 0]
    return FC_regressed


def control_y(y, control):
    """
    Controls y for confounding effects
    :param y: array containing the variable of interest for each participant, here FSIQ value 
    :param control: array containing the variable to cotnrol for (here age)
    :return: controlled y
    """
    if control.ndim == 1:
        control = control.reshape((-1, 1))
    reg = LinearRegression().fit(control, y.reshape((-1, 1)))
    iq_controlled = y - reg.coef_ * control[:, 0]
    return iq_controlled


def calc_losses(results, iqs):
    """
    Calculates the metrics that are reported
    :param results: predicted iq values
    :param iqs: actual iq values
    :returns: List containing result scores in 4 metrics
    """
    r, _ = pearsonr(iqs, results)
    mse = mean_squared_error(iqs, results)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(iqs, results)
    return [r, mse, rmse, mae]


def calc_all_result_metrics(y_pred, y, print_results=False):
    """
    Calculate and print the metrics for the positive and negative feature network
    :param y_pred (np array): predicted values
    :param y (np array): actual values
    :param print_results: boolean if the metric results should be printed. Default is False
    :returns: Numpy array containing result scores in 4 metrics
    """
    final_results = calc_losses(y_pred, y)
    if print_results:
        print(f"r={round(final_results[0], 3)}, mse={round(final_results[1], 3)},",
              f"rmse={round(final_results[2], 3)}, mae={round(final_results[3], 3)}")
    return final_results





