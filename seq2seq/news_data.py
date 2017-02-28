import os
import json
import re
import time
from collections import defaultdict
import numpy as np
import spacy

NLP = spacy.load('en')

def timer(function):
  '''Simple timer decorator.'''
  def wrapper(*args, **kwargs):
    start = time.time()
    result = function(*args, **kwargs)
    print('cpu time: {:.3f} s'.format(time.time() - start))
    return result
  return wrapper

def generator_(data_dir='../data', kind='politics-articles'):
  '''Generic generator to read lines from file.'''
  with open(os.path.join(data_dir, '{}.txt'.format(kind)), 'r') as f:
    for line in f:
      yield unicode(line.strip(), encoding='utf-8')

def data_generator(data_dir='../data'):
  '''Generate a tuple of article text, label and url.'''
  for data in zip(generator_(data_dir, 'politics-articles'), 
                  generator_(data_dir, 'politics-labels'), 
                  generator_(data_dir, 'politics-urls')):
    yield data

def read_data_in_memory():
  '''Read data directly into memory.'''
  data = []
  for datum in data_generator():
    data.append(datum)
  return data

def parse_doc(doc, keep_stops=True, min_sents=3, max_sents=10):
  '''Return text containing only lemmatized, alphanumeric tokens with POS tag.

  Args:
    doc: string
    keep_stop:  boolean, keep stop words
    min_sents:  minimum number of sentences to parse, else return ''
    max_sents: max number of sentences to return
  '''
  text = []
  sent_count = 0
  num_sents = len([sent for sent in doc.sents])
  if num_sents < min_sents:
    return ''
  for i, sent in enumerate(doc.sents):
    for token in sent:
      if not token.is_alpha:
        continue
      if not keep_stops and token.is_stop:
        continue
      else:   
        text.append('{}|{}'.format(token.lemma_, token.pos_))
    if i+1 == max_sents:
      break
  return ' '.join(text)

@timer
def parse_corpus(doc_list, keep_stops=True, min_sents=3, max_sents=10):
  '''Use NLP pipeline to parse a list of strings.'''
  parsed_corpus = []
  for doc in NLP.pipe(doc_list, batch_size=50, n_threads=4):
      parsed_corpus.append(
        parse_doc(doc, keep_stops=keep_stops, min_sents=min_sents, max_sents=max_sents)
      )
  return parsed_corpus

def write(filename, data_list):
  '''Simple utility for writing newline-separated file.'''
  with open(filename, 'w') as f:
    _ = [f.write('{}\n'.format(x)) for x in data_list]

@timer
def build_vocabulary(parsed_data, vocab_size=10000):
  '''Build the vocabulary.'''
  vocab_dict = defaultdict(int)
  for d in parsed_data:
    for word in d[0].split():
      vocab_dict[word] += 1
  vocab_list = [(k, v) for k,v in vocab_dict.iteritems()]
  vocab_list.sort(key=lambda x: -1*x[1])
  ranked_word_list = [x[0] for x in vocab_list[:(vocab_size-1)]]
  vocab_dict_rev = {}
  for ix, x in enumerate(ranked_word_list):
    vocab_dict_rev[x] = ix
    
  unknown_token = '<UNK>'
  ranked_word_list.append(unknown_token)
  vocab_dict_rev[unknown_token] = vocab_size+1

  return vocab_dict, vocab_dict_rev, ranked_word_list

def lookup_index(vocab_dict_rev, word):
  '''Returns the index for the input word.'''
  try:
    return vocab_dict_rev[word]
  except:
    return vocab_dict_rev['<UNK>']

def encode_corpus(parsed_data, vocab_dict_rev):
  '''Encode the article text.'''
  encoded_corpus = np.ndarray(len(parsed_data), dtype=list)
  for ix, d in enumerate(parsed_data):
    words = d[0].split(' ')
    idx = [lookup_index(vocab_dict_rev, word) for word in words]
    encoded_corpus[ix] = idx
  return encoded_corpus





