#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'policecar'

import os
import logging
import dateutil

import numpy as np
import pandas as pd

from sklearn.externals import joblib
from sklearn.feature_selection import SelectPercentile, chi2
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

def train_classifier( train_file='train/train-sample.csv', 
					  recompute_feats=False ):
	'''
	Module that reads stackoverflow data from a .csv file, 
	generates features, and trains a classifier.
	'''
	
	# custom variables
	DATA_DIR = "../data/"
	SUBMISSION_DIR = "../data/submission/"

	train_file_sample = "train/train-sample.csv"
	train_file_large = "train/train.csv"
	feats_file = 'train/feats.npz'
			
	# display progress logs on stdout
	logging.basicConfig( level=logging.INFO,
						 format='%(asctime)s %(levelname)s %(message)s' )
	log = logging.getLogger(__name__)
	
	log.info( "π: read and parse data" )
	data = pd.io.parsers.read_csv( os.path.join( DATA_DIR, train_file_sample ),
			converters = { "PostCreationDate": dateutil.parser.parse,
						   "OwnerCreationDate": dateutil.parser.parse } )
	
	log.info( "π: encode labels" )
	labels = data.pop('OpenStatus')
	lbl_map = { 'not a real question': 0, 'not constructive': 1, 'off topic': 2,
				'open': 3, 'too localized': 4 } # cf. required submission format
	labels = labels.map( lbl_map )
	y = labels.values
	
	if recompute_feats:
		log.info( "π: generate features" )
		X = features.fit_transform( data )

		log.info( 'π: save features to file' )
		np.savez( DATA_DIR + feats_file, X=X )
	else:
		log.info( 'π: load features from file' )
		npz_file = np.load( DATA_DIR + feats_file )
		X = npz_file['X']
		
	log.info( 'π: select features' )
	fselect = SelectPercentile( score_func=chi2, percentile=42 ) # !?
	# X = fselect.fit_transform( X, y )
		
	log.info( 'π: define classifiers' )
	priors = cu.get_priors( train_file_large )
	clf_lda = LDA( priors=priors )
	clf_rfc = RandomForestClassifier( n_estimators=50, verbose=2, n_jobs=-1, random_state=0, 
								  compute_importances=True, max_features=None ) #, criterion='entropy' )
	clf_gbc = GradientBoostingClassifier()
	
	log.info( 'π: fit LDA' )
	clf_lda.fit( X, y )
	# X = cld_lda.transform( X, y )
	log.info( 'π: fit RFC' )
	clf_rfc.fit( X, y )
	log.info( 'π: fit GBC' )
	clf_gbc.fit( X, y )
	
	log.info( "π: compute feature ranking for RFC" )
	importances = clf_rfc.feature_importances_
	std = np.std([ tree.feature_importances_ for tree in clf_rfc.estimators_ ], axis=0 )
	indices = np.argsort( importances )[::-1]
	for f in xrange( 13 ): # the top thirteen features
		print "%d. feature %d (%f)" % (f + 1, indices[f], importances[ indices[f] ])
	
	log.info( 'π: save classifiers' )
	np.savez( SUBMISSION_DIR+'cfy.npz', X=X, y=y, fselect=fselect )
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