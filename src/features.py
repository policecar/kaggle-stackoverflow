from __future__ import division
# -*- coding: utf-8 -*-
__author__ = 'policecar'

import re
import os
import features
import itertools

import numpy as np
import pandas as pd
import nltk
import nltk.stem.snowball as snowball

try:
	import IPython
	from IPython import embed
	debug = True
except ImportError:
	pass

# custom variables
RESOURCES_DIR = './resources/'

stop_words_file = 'en_stop_words.txt'
tags_file = 'tags.npz'
voc_file = 'voc.npz'

def compute_features( data_file=None, feature_file=None, label_file=None ):
	import logging
	import csv
	
	logging.basicConfig( level=logging.INFO,
						 format='%(asctime)s %(levelname)s %(message)s' )
	log = logging.getLogger(__name__)
	
	log.info( "π: load resources for feature generation" )
	load_resources()
	
	log.info( "π: generate features" )
	reader = csv.reader( open( os.path.join( DATA_DIR, data_file )))
	header = reader.next()
	writer = csv.writer( open( os.path.join( DATA_DIR, feature_file ), "w"), 
						 lineterminator="\n" )
	if label_file:
		label_file = open( os.path.join( DATA_DIR, label_file ), "w")
		def write_to_file( features, label ):
			writer.writerow( features )
			label_file.write( label + "\n" )
	else:
		def write_to_file( features, label ):
			writer.writerow( features )		
	
	c = 0
	for row in reader:
		if c % 1000 == 0: 
			log.info( "π: compute " + str(c) )
		row[1] = dateutil.parser.parse( row[1] )
		row[3] = dateutil.parser.parse( row[3] )
		sample = pd.Series( row, index=header )
		preprocess( sample )
		features = generate_features( sample )
		write_to_file( features, row[14] )
		c += 1


def load_resources():
	
	fid = open( os.path.join( RESOURCES_DIR, stop_words_file ))
	global stop_words
	stop_words = frozenset([ line.strip('\n') for line in fid ])
	fid.close()
	
	npz_file = np.load( os.path.join( RESOURCES_DIR, tags_file ))
	global tags
	tags = npz_file[ 'tags' ].item()
	tags = dict([ ( k, v.tolist() ) for k,v in tags.items() ])
	
	npz_file = np.load( os.path.join( RESOURCES_DIR, voc_file ))
	global voc
	voc = npz_file[ 'text' ].item()
	voc = dict([ ( k, v.tolist() ) for k,v in voc.items() ])


def preprocess( datum ):
	
	global precs
	precs = pd.Series()
	
	# preprocess body field
	# datum = datum.fillna( 'nan' )
	if datum['BodyMarkdown'] == '': datum['BodyMarkdown'] = 'nan'
	precs = precs.append( pd.Series({ "BodyTokens": 
		nltk.word_tokenize( datum['BodyMarkdown'] ) }))
	precs = precs.append( pd.Series( { "BodyStemmed": 
		pd.Series( precs[ "BodyTokens" ] ).apply( stem ) } ))
	# the next call takes way too long ( for now )
	# precs = precs.append( pd.Series({ "BodyTagged": 
	# 	precs['BodyTokens'].apply( nltk.pos_tag ) }))
	
	# preprocess title field
	# datum = datum.fillna( 'nan' )
	# if datum['Title'] == '': datum['Title'] = 'nan'
	precs = precs.append( pd.Series( { "TitleTokens": 
		nltk.word_tokenize( datum.get('Title') ) } ))
	precs = precs.append( pd.Series( { "TitleStemmed": 
		pd.Series( precs[ "TitleTokens" ] ).apply( stem ) } ))
	
	# assemble tags
	precs = precs.append( pd.Series( { "Tags": 
		pd.Series( datum[[ "Tag%d" % d for d in range(1,6) ]]
		.values.flatten() ).dropna().values } ))


def generate_features( datum ):
	'''
	This method copies a lot of structure from benhamner's features.py
	cf. https://github.com/benhamner/Stack-Overflow-Competition
	'''
	feature_names = [ "ReputationAtPostCreation"
					, "OwnerUndeletedAnswerCountAtPostTime"
					, "UserAge"
					, "TitleLength"
					, "BodyLength"
					, "TitleBodyRatio"
					, "NumTitleTokens"
					, "NumBodyTokens"
					, "TitleBodyTokenRatio"
					, "TitleBodyOverlap"
					, "AvgBodyTokenLength"
					, "ContainsCodeInsertion"
					, "NumCodeInsertions"
					, "NumUrls"
					, "NumNewlines"
					, "NumSentences"
					, "NewlineSentenceRatio"
					, "NewlineTokenRatio"
					, "TokenSentenceRatio"
					, "NumTitleStopwords"
					, "NumBodyStopwords"
					, "StopwordTokenRatio"
					, "TagsLength"
					, "AvgTagLength"
					, "NumTags"
					, "TagsOpen"
					, "TagsTooLocalized"
					, "TagsNotConstructive"
					, "TagsOffTopic"
					, "TagsNotARealQuestion"
					, "TitlesOpen"
					, "TitlesTooLocalized"
					, "TitlesNotConstructive"
					, "TitlesOffTopic"
					, "TitlesNotARealQuestion"
					# , "NumEntities"
					]
	
	global feats
	feats = pd.DataFrame( index=datum.index )
	feats = pd.Series()
	for name in feature_names:
		if name in datum:
			feats = feats.append( pd.Series({ name: float( datum[ name ] )}) )
		else:
			feats = feats.append( getattr( features, 
				camel_to_underscores( name ))( datum ) )
	# add combinations of all hand-crafted features to features
	more_feats = [ combine( pair ) for pair in list( itertools.combinations( feats.index, 2 )) ]
	feats = feats.append( more_feats )
	return feats


def combine( pair ):
	return pd.Series({ pair[0] + '_' + pair[1] : feats[ pair[0] ] * feats[ pair[1] ]  })

def user_age( datum ):
	return pd.Series({ "UserAge": 
		total_minutes( datum['PostCreationDate'] - datum['OwnerCreationDate'] ) })

def total_minutes( td ):
	return ( td.seconds / 60 ) + ( td.days * 24 * 60 )


def title_length( datum ):
    return pd.Series({ "TitleLength": len( datum['Title'] ) })

def body_length( datum ):
    return pd.Series({ "BodyLength": len( datum['BodyMarkdown'] ) })

def title_body_ratio( datum ):
	return pd.Series({ "TitleBodyRatio": 
		feats['TitleLength'] / feats['BodyLength'] })


def num_title_tokens( datum ):
	return pd.Series({ "NumTitleTokens": len( precs['TitleTokens'] ) })

def num_body_tokens( datum ):
	return pd.Series({ "NumBodyTokens": len( precs['BodyTokens'] )  })


def title_body_token_ratio( datum ):
	return pd.Series({ "TitleBodyTokenRatio": 
		feats['NumTitleTokens'] / feats['NumBodyTokens'] })


def title_body_overlap( datum ):
	return pd.Series({ "TitleBodyOverlap":
		len( set( precs['TitleStemmed'] ).intersection( precs['BodyStemmed'] )) })

def avg_body_token_length( datum ):
	return pd.Series({ "AvgBodyTokenLength":
		sum( pd.Series( precs['BodyTokens'] ).apply( len )) / len( precs['BodyTokens'] ) })

def contains_code_insertion( datum ): # this is a heuristics
	pattern = re.compile( r"\n\n    ", re.UNICODE )
	return pd.Series({ "ContainsCodeInsertion": 1 
		if pattern.search( datum['BodyMarkdown'] ) else 0 })

def num_code_insertions( datum ): # this is a heuristics
	pattern = re.compile( r"\n\n    ", re.UNICODE )
	return pd.Series({ "NumCodeInsertions": 
		len( pattern.findall( datum['BodyMarkdown'] )) })

def num_urls( datum ):
	pattern = re.compile( r"https?://", re.UNICODE )
	return pd.Series({ "NumUrls": len( pattern.findall( datum['BodyMarkdown'] )) })


def num_newlines( datum ):
	pattern = re.compile( r"\n", re.UNICODE )
	return pd.Series({ "NumNewlines": len( pattern.findall( datum['BodyMarkdown'] )) })

def num_sentences( datum ):
	return pd.Series({ "NumSentences": # is min 1 :)
		len( nltk.sent_tokenize( datum['BodyMarkdown'].strip() )) })

def newline_sentence_ratio( datum ):
	return pd.Series({ "NewlineSentenceRatio": 
		feats['NumNewlines'] / feats['NumSentences'] })

def newline_token_ratio( datum ):
	return pd.Series({ "NewlineTokenRatio": 
		feats['NumNewlines'] / feats['NumBodyTokens'] })

def token_sentence_ratio( datum ):
	return pd.Series({ "TokenSentenceRatio": 
		feats['NumBodyTokens'] / feats['NumSentences'] })


def num_title_stopwords( datum ):
	return pd.Series({ "NumTitleStopwords": 
		len( stop_words.intersection( precs['TitleTokens'] )) })

def num_body_stopwords( datum ):
	return pd.Series({ "NumBodyStopwords": 
		len( stop_words.intersection( precs['BodyTokens'] )) })

def stopword_token_ratio( datum ):
	return pd.Series({ "StopwordTokenRatio": 
		feats['NumBodyStopwords'] / feats['NumBodyTokens'] })

def tags_length( datum ):
	return pd.Series({ "TagsLength": len( " ".join( precs['Tags'] )) })

def avg_tag_length( datum ):
	return pd.Series({ "AvgTagLength":
		sum( pd.Series( precs['Tags'] ).apply( len )) / len( precs['Tags'] ) 
			if len( precs['Tags'] ) > 0 else 0 })

def num_tags( datum ):
    return pd.Series({ "NumTags": len( precs['Tags'] ) })


def tags_open( datum ):
	return pd.Series({ "TagsOpen":
		len( set( precs['Tags'] ).intersection( tags[ 'open' ] )) / len( tags[ 'open' ] ) })

def tags_too_localized( datum ):
	return pd.Series({ "TagsTooLocalize":
		len( set( precs['Tags'] ).intersection( tags[ 'too localized' ] )) 
		/ len( tags[ 'too localized' ] ) })
	

def tags_not_constructive( datum ):
	return pd.Series({ "TagsNotConstructive":
		len( set( precs['Tags'] ).intersection( tags[ 'not constructive' ] )) 
		/ len( tags[ 'not constructive' ] ) })

def tags_off_topic( datum ):
	return pd.Series({ "TagsOffTopic":
		len( set( precs['Tags'] ).intersection( tags[ 'off topic' ] )) 
		/ len( tags[ 'off topic' ] ) })

def tags_not_a_real_question( datum ):
	return pd.Series({ "TagsNotARealQuestion":
		len( set( precs['Tags'] ).intersection( tags[ 'not a real question' ] )) 
		/ len( tags[ 'not a real question' ]) })


def titles_open( datum ):
	return pd.Series({ "TitlesOpen":
		len( set( precs['TitleStemmed'] ).intersection( voc[ 'open' ] )) 
		/ len( voc[ 'open' ] ) })

def titles_too_localized( datum ):
	return pd.Series({ "TitlesTooLocalize":
		len( set( precs['TitleStemmed'] ).intersection( voc[ 'too localized' ] )) 
		/ len( voc[ 'too localized' ] ) })

def titles_not_constructive( datum ):
	return pd.Series({ "TitlesNotConstructive":
		len( set( precs['TitleStemmed'] ).intersection( voc[ 'not constructive' ] )) 
		/ len( voc[ 'not constructive' ] ) })

def titles_off_topic( datum ):
	return pd.Series({ "TitlesOffTopic":
		len( set( precs['TitleStemmed'] ).intersection( voc[ 'off topic' ] )) 
		/ len( voc[ 'off topic' ] ) })

def titles_not_a_real_question( datum ):
	return pd.Series({ "TitlesNotARealQuestion":
		len( set( precs['TitleStemmed'] ).intersection( voc[ 'not a real question' ] )) 
		/ len( voc[ 'not a real question' ] ) })


def num_entities( datum ):
	return pd.Series({ "NumEntities": 
		len( nltk.chunk.ne_chunk( precs['BodyTagged'] )) })


def stem( token ):
	token_stem = token.decode( 'utf-8', 'ignore' )
	try: # note: stemming includes lower casing
		token_stem = snowball.EnglishStemmer().stem( token_stem.lower() )
	except:
		pass
	return token_stem


def camel_to_underscores( name ):
    s1 = re.sub( '(.)([A-Z][a-z]+)', r'\1_\2', name )
    return re.sub( '([a-z0-9])([A-Z])', r'\1_\2', s1 ).lower()
