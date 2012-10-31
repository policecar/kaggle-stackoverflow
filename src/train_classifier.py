#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'policecar'

import os
import csv
import logging
import dateutil

import numpy as np
import pandas as pd

from sklearn.externals import joblib
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.lda import LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

import features
import competition_utilities as cu

try:
	import IPython
	from IPython import embed
	debug = True
except ImportError:
	pass

def train_classifier( train_file='train/train.csv', recompute_feats=False ): 
	'''
	Module that reads stackoverflow data from a .csv file, 
	generates features, and trains a classifier.
	'''
	
	# custom variables
	DATA_DIR = "../data/"
	SUBMISSION_DIR = "../data/submission/"
	# train_file = 'train/train-sample.csv'
	label_file = 'train/train-labels.csv'
	feature_file = 'train/train-feats.csv'
			
	# display progress logs on stdout
	logging.basicConfig( level=logging.INFO,
						 format='%(asctime)s %(levelname)s %(message)s' )
	log = logging.getLogger(__name__)
	
	if recompute_feats:
		features.compute_features( train_file, feature_file, label_file )
	
	log.info( 'π: load features from file' )
	X = pd.io.parsers.read_csv( os.path.join( DATA_DIR, feature_file ), header=None )
	X = X.as_matrix()

	log.info( "π: encode labels" )
	labels = pd.io.parsers.read_csv( os.path.join( DATA_DIR, label_file ), header=None )['X0']
	lbl_map = { 'not a real question': 0, 'not constructive': 1, 'off topic': 2,
				'open': 3, 'too localized': 4 } # cf. required submission format
	labels = labels.map( lbl_map )
	y = labels.values
	
	log.info( 'π: select features' )
	fselect = SelectPercentile( score_func=chi2, percentile=42 ) # !?
	# X = fselect.fit_transform( X, y )
	
	log.info( 'π: define classifiers' )
	priors = cu.get_priors( os.path.join( DATA_DIR, 'train/train.csv' ) )
	clf_lda = LDA( priors=priors )
	clf_rfc = RandomForestClassifier( n_estimators=50, verbose=2, n_jobs=-1, random_state=0, 
				compute_importances=True, max_features=None ) #, criterion='entropy' )
	clf_gbc = GradientBoostingClassifier()

	log.info( 'π: fit Random Forest' )
	clf_rfc.fit( X, y )

	log.info( "π: compute feature ranking for RFC" )
	importances = clf_rfc.feature_importances_
	std = np.std([ tree.feature_importances_ for tree in clf_rfc.estimators_ ], axis=0 )
	indices = np.argsort( importances )[::-1]
	for f in xrange( 13 ): # the top thirteen features
		print "%d. feature %d (%f)" % (f + 1, indices[f], importances[ indices[f] ])

	log.info( "π: standardize and normalize features" )
	standardizer = StandardScaler( copy=False ).fit( X, y )
	standardizer.transform( X, y )	# in-place
	normalizer = Normalizer( copy=False, norm='l2' ).fit( X, y ) # 'l1'
	normalizer.transform( X, y )	# in-place
	
	log.info( 'π: fit Linear Discriminant Analysis' )
	clf_lda.fit( X, y )
	# X = cld_lda.transform( X, y )
	log.info( 'π: fit Gradient Boosting' )
	clf_gbc.fit( X, y )
	
	log.info( 'π: save classifiers' )
	np.savez( SUBMISSION_DIR+'cfy.npz', X=X, y=y, fselect=fselect, 
				standardizer=standardizer, normalizer=normalizer )
	joblib.dump( clf_lda, SUBMISSION_DIR + 'clf_lda.pkl', compress=9 )
	joblib.dump( clf_rfc, SUBMISSION_DIR + 'clf_rfc.pkl', compress=9 )
	joblib.dump( clf_gbc, SUBMISSION_DIR + 'clf_gbc.pkl', compress=9 )


if __name__ == "__main__":
	import sys
	try:
		b = True if sys.argv[1] == 'True' else False
		train_classifier( recompute_feats=b )
	except:
		train_classifier()