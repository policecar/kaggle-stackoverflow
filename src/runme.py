#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'policecar'

import train_classifier as train
import predict_class as predict

# train classifier(s)
train.train_classifier( recompute_feats=True )

# predict classes for data points in test_file
predict.predict_class( test_file='test/public_leaderboard.csv', 
					   recompute_feats=True )