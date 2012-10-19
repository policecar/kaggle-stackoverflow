# -*- coding: utf-8 -*-
__author__ = 'policecar'

import os
import logging
import csv

import numpy as np
import pandas as pd

import nltk
import nltk.stem.snowball as snowball

import competition_utilities as cu

DATA_DIR = '../data/'
RESOURCES_DIR = './resources/'
file_name = 'train.csv'

logging.basicConfig( level=logging.INFO,
					format='%(asctime)s %(levelname)s %(message)s' )
log = logging.getLogger(__name__)

log.info( "π: read data" )
header = cu.get_header( file_name )
open_status = [ r[14] for r in cu.get_reader( file_name ) ]

def generate_tags():
	log.info( "π: read tags" )
	tags = [ r[8:13] for r in cu.get_reader( file_name ) ]
	
	log.info( "π: process tags" )
	res = {}
	for st in pd.Series( open_status ).unique():
		# res.setdefault( st, set() )
		res.setdefault( st, [] )

	for i,x in enumerate( open_status ):
		# res[x] = res[x].union( tags[i] )
		res[x].extend( tags[i] )

	res = dict([ ( k, pd.Series( v ).unique() ) for k,v in res.items() ])

	log.info( "π: save tags" )
	np.savez( RESOURCES_DIR + 'tags.npz', tags=res )
	del tags


def generate_titles():
	log.info( "π: read text" )
	# text = [ r[6:8] for r in cu.get_reader( file_name ) ]
	text = [ r[6] for r in cu.get_reader( file_name ) ]

	log.info( "π: tokenize text" )
	text = [ nltk.word_tokenize(t) for t in text ]

	log.info( "π: stem tokens" )
	text = [ pd.Series( t ).apply( stem ) for t in text ]

	log.info( "π: to lower case" )
	text = [ pd.Series( t ).apply( lower ) for t in text ]

	log.info( "π: process text" )
	res = {}
	for st in pd.Series( open_status ).unique():
		res.setdefault( st, [] )

	for i,x in enumerate( open_status ):
		res[x].extend( text[i] )

	log.info( "π: uniquify text" )
	res = dict([ ( k, pd.Series( v ).unique() ) for k,v in res.items() ])

	log.info( "π: save vocabulary" )
	np.savez( RESOURCES_DIR + 'titles.npz', text=res )


def stem( token ):
	token_stem = token.decode( 'utf-8', 'ignore' )
	try:
		token_stem = snowball.EnglishStemmer().stem( token_stem )
	except:
		pass
	return token_stem

def lower( token ):
	return token.lower()


if __name__ == "__main__":
	import sys
	try:
		if sys.argv[1] == 'tags':
			generate_tags()
	except:
		generate_titles()
