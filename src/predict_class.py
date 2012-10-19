#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'policecar'

import logging
import csv
import os
import dateutil

import numpy as np
import pandas as pd

from sklearn.externals import joblib

import competition_utilities as cu
import features

try:
	import IPython
	from IPython import embed
	debug = True
except ImportError:
	pass

def predict_class( test_file='test/public_leaderboard.csv', 
				   recompute_feats=False ):
	'''
	Module that predicts class probabilities for test data 
	from a .csv file or precomputed feature vectors.
	'''
	
	# custom variables
	DATA_DIR = "../data/"
	SUBMISSION_DIR = "../data/submission/"

	train_file_sample = "train/train-sample.csv"
	train_file_large = "train/train.csv"
	test_feats_file = 'test/public_leaderboard_feats.npz' #'test_feats.npz'
	output_file = 'predictions.csv'
	
	logging.basicConfig( level=logging.INFO,
						 format='%(asctime)s %(levelname)s %(message)s' )
	log = logging.getLogger(__name__)
	
	if recompute_feats:
		log.info( "π: parse test data" )
		test_data = pd.io.parsers.read_csv( os.path.join( DATA_DIR, test_file ),
			converters = { "PostCreationDate": dateutil.parser.parse,
						   "OwnerCreationDate": dateutil.parser.parse } )
		log.info( "π: extract features" )
		X_test = features.transform( test_data )
	else:
		log.info( "π: load test features" )
		npz_file = np.load( DATA_DIR + test_feats_file )
		X_test = npz_file[ 'X' ]

	log.info( "π: load classifier" )
	npz_file = np.load( SUBMISSION_DIR + 'cfy.npz' )
	clf_lda = joblib.load( SUBMISSION_DIR + 'clf_lda.pkl' )
	clf_rfc = joblib.load( SUBMISSION_DIR + 'clf_rfc.pkl' )
	clf_gbc = joblib.load( SUBMISSION_DIR + 'clf_gbc.pkl' )

	log.info( 'π: perform feature selection' )
	fselect = npz_file[ 'fselect' ].item()	
	# X_test = fselect.transform( X_test )
	
	log.info( "π: predict class membership probabilities" )
	# X_test = clf_lda.transform( X_test ) 
	y_lda = clf_lda.predict_proba( X_test )
	y_rfc = clf_rfc.predict_proba( X_test )
	y_gbc = clf_gbc.predict_proba( X_test )

	y_pred = ( y_rfc + y_gbc ) / 2.0

	log.info( "π: calculate priors and update posteriors" )
	new_priors = cu.get_priors( train_file_large )
	old_priors = cu.get_priors( train_file_sample )
	y_pred = cu.cap_and_update_priors( old_priors, y_pred, new_priors, 0.001 )
	
	y_pred = ( y_pred + y_lda ) / 2.0
	
	log.info( "π: write predictions to file" )
	writer = csv.writer( open( os.path.join( SUBMISSION_DIR, output_file ), 
							"w"), lineterminator="\n" )
	writer.writerows( y_pred )


if __name__ == "__main__":
	import sys
	try:
		b = True if sys.argv[1] == 'True' else False
		predict_class()( recompute_feats=b )
	except:
		predict_class()