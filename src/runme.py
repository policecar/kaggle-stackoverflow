#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'policecar'

import train_classifier as train
import predict_class as predict

train.train_classifier( recompute_feats=True )

predict.predict_class( test_file='test/private_leaderboard.csv', 
	recompute_feats=True )
