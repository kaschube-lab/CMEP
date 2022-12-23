"""
=============================================================================================================
Project: How much data do we need? Lower bounds of brain activation states to predict human cognitive ability
=============================================================================================================
Description
-----------
Select specific time frames from the resting-state time series.

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
from scipy.signal import argrelextrema


def create_FC_from_tfs(raw_signal, n_frames=44, selection_type='HiCo'):
    """
    Select time frames of a specific type from the raw signal time series. 
    :param raw_signal (np array): raw bold signal shape n_participants x n_nodes x T
    :param n_frames (int): how many frames to select from the time series. 
    :param selection_type (str): which time frames to select
    :return: functional connectivity for each participant
    """
    n_participants, n_nodes, T = raw_signal.shape
    FC_tfs = np.zeros((n_participants, n_nodes, n_nodes))
    for i, raw_sample in enumerate(raw_signal):
        selected_tfs = select_time_frames(raw_sample, n_frames, selection_type)
        FC_tfs[i] = np.corrcoef(raw_sample[:, selected_tfs])
    return FC_tfs


def select_time_frames(raw_sample, n_frames, selection_type):
    """
    Select time frames of a specific type from the raw signal time series. 
    :param raw_sample (np array): raw bold signal shape n_nodes x T
    :param n_frames (int): how many frames to select from the time series. 
    :param selection_type (str): which time frames to select
    :return: np array with selected time frames. 
    """
    rss = calc_rss(raw_sample)
    selected_tfs = get_tfs(rss, n_frames, selection_type)
    return selected_tfs



def get_tfs(rss_sample, n_frames, selection_type):
    """
    :param rss_sample (np array): raw bold signal for one participant with length T
    :param n_frames (int): how many frames to select from the time series
    :param selection_type (str): which time frames to select
    :return: np array with selected time frames for the participant
    """
    if selection_type == 'HiCo':
        return np.argsort(rss_sample)[::-1][:n_frames]
    elif selection_type == 'LoCo':
        return np.argsort(rss_sample)[:n_frames]
    elif selection_type == 'MxCo':
        HiCo = np.argsort(rss_sample)[::-1][:n_frames]
        (all_max,) = argrelextrema(rss_sample, np.greater)
        return np.intersect1d(HiCo, all_max)
    elif selection_type == 'MnCo':
        LoCo = np.argsort(rss_sample)[:n_frames]
        (all_min,) = argrelextrema(rss_sample, np.less)
        return np.intersect1d(LoCo, all_min)
    elif selection_type == 'Mx':
        (all_max,) = argrelextrema(rss_sample, np.greater)
        max_high_to_low = all_max[np.argsort(rss_sample[all_max])[::-1]]
        return max_high_to_low[:n_frames]
    elif selection_type == 'Mn':
        (all_min,) = argrelextrema(rss_sample, np.less)
        min_low_to_high = all_min[np.argsort(rss_sample[all_min])]
        return min_low_to_high[:n_frames]
    elif selection_type == 'random':
        T = len(rss_sample)
        return np.random.choice(T, n_frames)
    else:
        raise ValueError('Selection type has to be HiCo, LoCo, MxCo, MnCo, Mx, Mn or random.')


def calc_rss(raw_sample):
    """
    Calculate the root sum of squares (RSS) from the raw signal. 
    :param raw_sample (np array): raw bold signal shape n_nodes x T
    :return: rss time series for each participant (np array)
    """
    _, T = raw_sample.shape
    raw_stand = (raw_sample - np.nanmean(raw_sample, axis=-1, keepdims=True)) / np.nanstd(raw_sample, axis=-1, keepdims=True)
    rss = np.zeros(T)
    for t in range(T):
        co_fluct = np.outer(raw_stand[:, t], raw_stand[:, t])
        rss[t] = np.sqrt(np.sum(co_fluct.ravel() ** 2))
    return rss



