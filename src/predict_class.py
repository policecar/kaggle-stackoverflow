from __future__ import division
# -*- coding: utf-8 -*-
__author__ = 'policecar'

import logging
import csv
import os
import dateutil
from collections import Counter

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
	train_file_all = "train/train.csv"
	test_file = 'test/private_leaderboard.csv'
	feature_file = 'test/private_leaderboard-feats.csv'
	output_file = 'predictions.csv'


	logging.basicConfig( level=logging.INFO,
						 format='%(asctime)s %(levelname)s %(message)s' )
	log = logging.getLogger(__name__)
	
	if recompute_feats:
		# features.compute_features( 'test/test.csv', 'test/test-feats.csv' )
		features.compute_features( test_file, feature_file )
		
	log.info( 'π: load features from file' )
	X_test = pd.io.parsers.read_csv( os.path.join( DATA_DIR, feature_file ), header=None )
	X_test = X_test.as_matrix()
	
	log.info( "π: load classifier" )
	npz_file = np.load( SUBMISSION_DIR + 'cfy.npz' )
	clf_lda = joblib.load( SUBMISSION_DIR + 'clf_lda.pkl' )
	clf_rfc = joblib.load( SUBMISSION_DIR + 'clf_rfc.pkl' )
	clf_gbc = joblib.load( SUBMISSION_DIR + 'clf_gbc.pkl' )
	
	log.info( "π: load standardizer, normalizer" )
	standardizer = npz_file[ 'standardizer' ].item()
	normalizer = npz_file[ 'normalizer' ].item()

	# log.info( 'π: perform feature selection' )
	# fselect = npz_file[ 'fselect' ].item()	
	# X_test = fselect.transform( X_test )

	log.info( "π: Random Forest predictions" )
	y_rfc = clf_rfc.predict_proba( X_test )

	log.info( "π: standardize and normalize test features" )
	standardizer.transform( X_test ) # in-place
	normalizer.transform( X_test )	 # in-place
	
	log.info( "π: LDA and GBC class membership predictions" )
	# X_test = clf_lda.transform( X_test ) 
	y_lda = clf_lda.predict_proba( X_test )
	y_gbc = clf_gbc.predict_proba( X_test )

	y_pred = ( y_rfc + y_gbc ) / 2.0

	log.info( "π: calculate priors and update posteriors" )
	new_priors = cu.get_priors( train_file_all )
	closed_reasons = pd.io.parsers.read_csv( os.path.join( DATA_DIR, train_labels ), header=None )['X0']
	closed_reason_counts = Counter( closed_reasons )
	reasons = sorted( closed_reason_counts.keys() )
	total = len( closed_reasons )
	old_priors = [ closed_reason_counts[ reason ] / total for reason in reasons ]
	y_pred = cu.cap_and_update_priors( old_priors, y_pred, new_priors, 0.001 )
	
	y_pred = ( 2 * y_pred + y_lda ) / 3.0
	log.info( "π: write predictions to file" )
	writer = csv.writer( open( os.path.join( SUBMISSION_DIR, output_file ), "w"), 
				lineterminator="\n" )
	writer.writerows( y_pred )


if __name__ == "__main__":
	import sys
	try:
		b = True if sys.argv[1] == 'True' else False
		predict_class()( recompute_feats=b )
	except:
		predict_class()