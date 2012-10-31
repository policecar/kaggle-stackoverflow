from __future__ import division
# -*- coding: utf-8 -*-

import numpy as np

def multiclass_log_loss( y_true, y_pred, eps=1e-15 ):
    """
    Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss

    idea from this post:
    http://www.kaggle.com/c/emc-data-science/forums/t/2149/
        is-anyone-noticing-difference-betwen-validation-and-leaderboard-error/12209#post12209

	implementation adopted from ephes, cf.
	https://github.com/ephes/scikit-learn/blob/multiclass_log_loss/sklearn/metrics/metrics.py

    Parameters
    ----------
    y_true : array, shape = [ n_samples ]
    y_pred : array, shape = [ n_samples, n_classes ]

    Returns
    -------
    loss : float
    """    
    predictions = np.clip( y_pred, eps, 1 - eps )
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    groundtruth = np.zeros( y_pred.shape )
    rows = groundtruth.shape[ 0 ]
    groundtruth[ np.arange(rows), y_true.astype(int) ] = 1
    logloss = np.sum( groundtruth * np.log( predictions ))
    return -1.0 / rows * logloss