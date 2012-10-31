#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'policecar'

import sys
import csv
import numpy as np
import pandas as pd

import metrics

path_to_true_y = '../data/test/y-true.csv'
path_to_pred_y = '../data/submission/predictions.csv'

reader = csv.reader(open( path_to_true_y ), delimiter=',' )
y_true = []
for row in reader:
	y_true.append( pd.Series(row).apply( float ))

reader = csv.reader(open( path_to_pred_y ), delimiter=',' )
y_pred = []
for row in reader:
	y_pred.append( pd.Series(row).apply( float ))

print  metrics.multiclass_log_loss( np.asarray( y_true ) , np.asarray( y_pred ))
