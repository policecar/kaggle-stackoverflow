from __future__ import division
# -*- coding: utf-8 -*-
__author__ = 'policecar'

import re
import os
import logging
import features

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

# get logger
logging.basicConfig( level=logging.INFO,
					 format='%(asctime)s %(levelname)s %(message)s' )
log = logging.getLogger(__name__)

def preprocess( data ):
	
	log.info( "π: load stop words" )
	fid = open( os.path.join( RESOURCES_DIR, stop_words_file ))
	global stop_words
	stop_words = frozenset([ line.strip('\n') for line in fid ])
	fid.close()
	
	log.info( "π: load tags file" )
	npz_file = np.load( os.path.join( RESOURCES_DIR, tags_file ))
	global tags
	tags = npz_file[ 'tags' ].item()
	tags = dict([ ( k, v.tolist() ) for k,v in tags.items() ])

	log.info( "π: load voc file" )
	npz_file = np.load( os.path.join( RESOURCES_DIR, voc_file ))
	global voc
	voc = npz_file[ 'text' ].item()
	voc = dict([ ( k, v.tolist() ) for k,v in voc.items() ])
	
	global precs
	precs = pd.DataFrame( index=data.index )
	
	# body
	log.info( "π: tokenize body" )
	data.BodyMarkdown = data.BodyMarkdown.fillna( 'nan' )
	# pattern = re.compile( r"\b\w\w+\b", re.UNICODE )
	# precs = precs.join( pd.DataFrame.from_dict({ "BodyTokens": data.BodyMarkdown.fillna('').apply( pattern.findall ) }) )
	precs = precs.join( pd.DataFrame.from_dict({ "BodyTokens": data.BodyMarkdown.apply( nltk.word_tokenize ) }))
	log.info( "π: stem body tokens" )
	precs = precs.join( pd.DataFrame.from_dict({ "BodyStemmed":
						[ pd.Series( t ).apply( stem ) for t in precs.BodyTokens ] }))	
	# the next line takes way too long ( for now )
	# precs = precs.join( pd.DataFrame.from_dict({ "BodyTagged": precs.BodyTokens.apply( nltk.pos_tag ) }))

	# title
	log.info( "π: tokenize title" )
	data.Title = data.Title.fillna( 'nan' )
	precs = precs.join( pd.DataFrame.from_dict({ "TitleTokens": data.Title.apply( nltk.word_tokenize ) }))
	log.info( "π: stem title tokens" )
	precs = precs.join( pd.DataFrame.from_dict({ "TitleStemmed":
						[ pd.Series( t ).apply( stem ) for t in precs.TitleTokens ] }))	
	
	# tags
	log.info( "π: assemble tags" )
	precs = precs.join( pd.DataFrame.from_dict({ "Tags": [ pd.Series( row ).dropna().values
						for row in ( data[[ "Tag%d" % d for d in range(1,6) ]].values ) ] }))
	

def fit_transform( data ):
	
	preprocess( data )
	return generate_features( data )

def transform( data ):
	# in this case the same as fit_transform, kept for structural consistency
	preprocess( data )
	return generate_features( data )


def generate_features( data ):
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
	feats = pd.DataFrame( index=data.index )
	for name in feature_names:
		log.info( "π: compute " + name )
		if name in data:
			feats = feats.join( data[ name ] )
		else:
			feats = feats.join( getattr( features,
				camel_to_underscores( name ))( data ))
	return feats

def user_age( data ):
    return pd.DataFrame.from_dict({ "UserAge": ( data[ "PostCreationDate" ]
		- data[ "OwnerCreationDate" ]).apply(lambda x: total_minutes( x ) )})

def total_minutes( td ):
	return ( td.seconds / 60 ) + ( td.days * 24 * 60 )


def title_length( data ):
    return data.Title.apply( len )

def body_length( data ):
    return data.BodyMarkdown.apply( len )

def title_body_ratio( data ):
	return pd.DataFrame.from_dict({ "TitleBodyRatio":
	 								feats.Title / feats.BodyMarkdown })


def num_title_tokens( data ):
	return pd.DataFrame.from_dict({ "NumTitleTokens":
									[ len(t) for t in precs.TitleTokens ] })

def num_body_tokens( data ):
	return pd.DataFrame.from_dict({ "NumBodyTokens":
									[ len(t) for t in precs.BodyTokens ] })

def title_body_token_ratio( data ):
	return pd.DataFrame.from_dict({ "TitleBodyTokenRatio":
									feats.NumTitleTokens / feats.NumBodyTokens })


def title_body_overlap( data ):
	return pd.DataFrame.from_dict({ "TitleBodyOverlap":
			[ len( set( precs.TitleStemmed[i]).intersection( v ))
			  for i,v in enumerate( precs.BodyStemmed ) ] })

def avg_body_token_length( data ):
	return pd.DataFrame.from_dict({ "AvgBodyTokenLength":
			[ sum( pd.Series(t).apply( len )) / len(t) for t in precs.BodyTokens ] })

def contains_code_insertion( data ): # this is a heuristics
	pattern = re.compile( r"\n\n    ", re.UNICODE )
	return pd.DataFrame.from_dict({ "ContainsCodeInsertion":
			[ 1 if pattern.search( t ) else 0 for t in data.BodyMarkdown ] })

def num_code_insertions( data ): # this is a heuristics
	pattern = re.compile( r"\n\n    ", re.UNICODE )
	return pd.DataFrame.from_dict({ "NumCodeInsertions":
			[ len( pattern.findall( t ) ) for t in data.BodyMarkdown ] })

def num_urls( data ):
	pattern = re.compile( r"https?://", re.UNICODE )
	return pd.DataFrame.from_dict({ "NumUrls":
			[ len( pattern.findall( t ) ) for t in data.BodyMarkdown ] })


def num_newlines( data ):
	pattern = re.compile( r"\n", re.UNICODE )
	return pd.DataFrame.from_dict({ "NumNewlines":
			[ len( pattern.findall( t ) ) for t in data.BodyMarkdown ] })

def num_sentences( data ):
	return pd.DataFrame({ "NumSentences": # is min 1 :)
			 [ len( nltk.sent_tokenize( t.strip() )) for t in data.BodyMarkdown ] })

def newline_sentence_ratio( data ):
	return pd.DataFrame({ "NewlineSentenceRatio":
						  feats.NumNewlines / feats.NumSentences })

def newline_token_ratio( data ):
	return pd.DataFrame({ "NewlineTokenRatio":
						  feats.NumNewlines / feats.NumBodyTokens })

def token_sentence_ratio( data ):
	return pd.DataFrame({ "TokenSentenceRatio": 
						  feats.NumBodyTokens / feats.NumSentences })


def num_title_stopwords( data ):
	return pd.DataFrame.from_dict({ "NumTitleStopwords":
			[ len( stop_words.intersection(t) ) for t in precs.TitleTokens ] })

def num_body_stopwords( data ):
	return pd.DataFrame.from_dict({ "NumBodyStopwords":
			[ len( stop_words.intersection(t) ) for t in precs.BodyTokens ] })

def stopword_token_ratio( data ):
	return pd.DataFrame.from_dict({ "StopwordTokenRatio":
									feats.NumBodyStopwords / feats.NumBodyTokens })

def tags_length( data ):
	return pd.DataFrame.from_dict({ "TagsLength":
									[ len( " ".join( t )) for t in precs.Tags ] })

def avg_tag_length( data ):
	return pd.DataFrame.from_dict({ "AvgTagLength":
			[ sum( pd.Series(t).apply( len )) / len(t) if len(t) > 0 else 0 for t in precs.Tags ] })

def num_tags( data ):
    return pd.DataFrame.from_dict({ "NumTags": [ len( t ) for t in precs.Tags ] })


def tags_open( data ):
	return pd.DataFrame.from_dict({ "TagsOpen":
			[ len( set( t ).intersection( tags[ 'open' ] )) / len( tags[ 'open' ] )
				for t in precs.Tags ] })

def tags_too_localized( data ):
	return pd.DataFrame.from_dict({ "TagsTooLocalize":
			[ len( set( t ).intersection( tags[ 'too localized' ] )) 
				/ len( tags[ 'too localized' ] ) for t in precs.Tags ] })

def tags_not_constructive( data ):
	return pd.DataFrame.from_dict({ "TagsNotConstructive":
			[ len( set( t ).intersection( tags[ 'not constructive' ] )) 
				/ len( tags[ 'not constructive' ] ) for t in precs.Tags ] })

def tags_off_topic( data ):
	return pd.DataFrame.from_dict({ "TagsOffTopic":
			[ len( set( t ).intersection( tags[ 'off topic' ] )) / len( tags[ 'off topic' ] )
				for t in precs.Tags ] })

def tags_not_a_real_question( data ):
	return pd.DataFrame.from_dict({ "TagsNotARealQuestion":
			[ len( set( t ).intersection( tags[ 'not a real question' ] )) 
				/ len( tags[ 'not a real question' ] ) for t in precs.Tags ] })


def titles_open( data ):
	return pd.DataFrame.from_dict({ "TitlesOpen":
			[ len( set( t ).intersection( voc[ 'open' ] )) / len( voc[ 'open' ] )
				for t in precs.TitleStemmed ] })

def titles_too_localized( data ):
	return pd.DataFrame.from_dict({ "TitlesTooLocalize":
			[ len( set( t ).intersection( voc[ 'too localized' ] )) 
				/ len( voc[ 'too localized' ] ) for t in precs.TitleStemmed ] })

def titles_not_constructive( data ):
	return pd.DataFrame.from_dict({ "TitlesNotConstructive":
			[ len( set( t ).intersection( voc[ 'not constructive' ] )) 
				/ len( voc[ 'not constructive' ] ) for t in precs.TitleStemmed ] })

def titles_off_topic( data ):
	return pd.DataFrame.from_dict({ "TitlesOffTopic":
			[ len( set( t ).intersection( voc[ 'off topic' ] )) / len( voc[ 'off topic' ] )
				for t in precs.TitleStemmed ] })

def titles_not_a_real_question( data ):
	return pd.DataFrame.from_dict({ "TitlesNotARealQuestion":
			[ len( set( t ).intersection( voc[ 'not a real question' ] )) 
				/ len( voc[ 'not a real question' ] ) for t in precs.TitleStemmed ] })


def num_entities( data ):
	return pd.DataFrame.from_dict({ "NumEntities":
			[ len( nltk.chunk.ne_chunk( t )) for t in precs.BodyTagged ] })


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
