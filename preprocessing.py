# this file preprocesses the data from NeuLab TED talk translation dataset
# (https://github.com/neulab/word-embeddings-for-nmt) for use with the
# transformer model in transformer.py
# it loads the Portuguese to English dataset, tokenizes them with a subword 
# tokenizer, and filters out any sentence with >40 tokens

import tensorflow_datasets as tfds
import tensorflow as tf
from constants import *

# returns training set, validation set (unprocessed)
def load_data():
  examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                                as_supervised=True)

  train_examples, val_examples = examples['train'], examples['validation']
  return train_examples, val_examples

# returns tokenizers for english and portuguese
def get_tokenizers(train_examples):
  tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
      (en.numpy() for pt, en in train_examples), target_vocab_size=2**13)

  tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
      (pt.numpy() for pt, en in train_examples), target_vocab_size=2**13)
  return tokenizer_en, tokenizer_pt

# demo english tokenizer
def demo_english_tokenizer(tokenizer_en, tokenizer_pt):
  sample_string = 'Transformer is awesome.'
  tokenized_string = tokenizer_en.encode(sample_string)
  print ('Tokenized string is {}'.format(tokenized_string))

  original_string = tokenizer_en.decode(tokenized_string)
  print ('The original string: {}'.format(original_string))

  assert original_string == sample_string

  for ts in tokenized_string:
    print ('{} ----> {}'.format(ts, tokenizer_en.decode([ts])))



# pt and en are strings
def tf_encode_wrapper(tokenizer_en, tokenizer_pt):
  # tokenize the data for a single portuguese sentence and its english translation
  def encode(pt, en):
    pt = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(
        pt.numpy()) + [tokenizer_pt.vocab_size+1]
    en = [tokenizer_en.vocab_size] + tokenizer_en.encode(
        en.numpy()) + [tokenizer_en.vocab_size+1]
    return pt, en
  
  # tensorflow function for tokenizing a single pair of (portuguese, english) text
  # we need this to be a tensorflow function so that we can use tensorflow's map function
  def tf_encode(pt, en):
    result_pt, result_en = tf.py_function(encode, [pt, en], [tf.int64, tf.int64])
    result_pt.set_shape([None])
    result_en.set_shape([None])

    return result_pt, result_en

  return tf_encode

# tensorflow function that filters out tokens longer than max_length
def filter_max_length(x, y, max_length=MAX_LENGTH):
  return tf.logical_and(tf.size(x) <= max_length,
                        tf.size(y) <= max_length)

# this is the primary function in this file.
# use load_data and get_tokenizers to get the arguments for this function
def preprocess_dataset(train_examples, val_examples, tokenizer_en, tokenizer_pt):
  # the function we'll apply to every element of the dataset
  tf_encode = tf_encode_wrapper(tokenizer_en, tokenizer_pt)
  train_dataset = train_examples.map(tf_encode)
  train_dataset = train_dataset.filter(filter_max_length)
  # cache the dataset to memory to get a speedup while reading from it.
  train_dataset = train_dataset.cache()
  train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)
  train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

  val_dataset = val_examples.map(tf_encode)
  val_dataset = val_dataset.filter(filter_max_length).padded_batch(BATCH_SIZE)

  return train_dataset, val_dataset

# does all the work without giving you the tokenizers or the initial data
def get_preprocessed_dataset():
  train_examples, val_examples = load_data()
  tokenizer_en, tokenizer_pt = get_tokenizers(train_examples)
  return preprocess_dataset(train_examples, val_examples, tokenizer_en, tokenizer_pt)