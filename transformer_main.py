#https://www.tensorflow.org/tutorials/text/transformer
import time

import tensorflow as tf
import matplotlib.pyplot as plt

from constants import *
# Preprocess the data (preprocessing.py)
from preprocessing import load_data
from preprocessing import get_tokenizers
from preprocessing import preprocess_dataset
from transformer_model import Transformer
from create_masks import create_masks
from transformer_utils import get_new_optimizer
from transformer_utils import get_checkpoint_object
from transformer_utils import get_checkpoint_manager
from transformer_utils import translate

def train_new_transformer():
  train_examples, val_examples = load_data()
  tokenizer_en, tokenizer_pt = get_tokenizers(train_examples)
  train_dataset, val_dataset = preprocess_dataset(train_examples, val_examples, tokenizer_en, tokenizer_pt)

  input_vocab_size = tokenizer_pt.vocab_size + 2
  target_vocab_size = tokenizer_en.vocab_size + 2

  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction='none')

  def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

  train_loss = tf.keras.metrics.Mean(name='train_loss')
  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='train_accuracy')

  transformer = Transformer(NUM_LAYERS, D_MODEL, NUM_HEADS, DFF,
                            input_vocab_size, target_vocab_size, 
                            pe_input=input_vocab_size, 
                            pe_target=target_vocab_size,
                            rate=DROPOUT_RATE)

  optimizer = get_new_optimizer()
  ckpt = get_checkpoint_object(transformer, optimizer)
  ckpt_manager = get_checkpoint_manager(ckpt)

  # if a checkpoint exists, restore the latest checkpoint.
  if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!')

  # The @tf.function trace-compiles train_step into a TF graph for faster
  # execution. The function specializes to the precise shape of the argument
  # tensors. To avoid re-tracing due to the variable sequence lengths or variable
  # batch sizes (the last batch is smaller), use input_signature to specify
  # more generic shapes.

  train_step_signature = [
      tf.TensorSpec(shape=(None, None), dtype=tf.int64),
      tf.TensorSpec(shape=(None, None), dtype=tf.int64),
  ]

  @tf.function(input_signature=train_step_signature)
  def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
    
    with tf.GradientTape() as tape:
      predictions, _ = transformer(inp, tar_inp, 
                                  True, 
                                  enc_padding_mask, 
                                  combined_mask, 
                                  dec_padding_mask)
      loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    
    train_loss(loss)
    train_accuracy(tar_real, predictions)

  print("Beginning first epoch")
  for epoch in range(EPOCHS):
    start = time.time()
    
    train_loss.reset_states()
    train_accuracy.reset_states()
    
    # inp -> portuguese, tar -> english
    for (batch, (inp, tar)) in enumerate(train_dataset):
      train_step(inp, tar)
      
      if batch % 50 == 0:
        print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
            epoch + 1, batch, train_loss.result(), train_accuracy.result()))
        
    if (epoch + 1) % 5 == 0:
      ckpt_save_path = ckpt_manager.save()
      print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                          ckpt_save_path))
      
    print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                  train_loss.result(), 
                                                  train_accuracy.result()))

    print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

  print("Training finished, saving model...")
  transformer.save_weights(MODEL_SAVE_PATH)
  print("Model saved in {}".format(MODEL_SAVE_PATH))

  translate("este é um problema que temos que resolver.", tokenizer_en, tokenizer_pt, transformer)
  print("Real translation: this is a problem we have to solve .")

  translate("os meus vizinhos ouviram sobre esta ideia.", tokenizer_en, tokenizer_pt, transformer)
  print("Real translation: and my neighboring homes heard about this idea .")

  translate("vou então muito rapidamente partilhar convosco algumas histórias de algumas coisas mágicas que aconteceram.", tokenizer_en, tokenizer_pt, transformer)
  print("Real translation: so i 'll just share with you some stories very quickly of some magical things that have happened .")

  translate("este é o primeiro livro que eu fiz.", tokenizer_en, tokenizer_pt, transformer, plot='decoder_layer4_block2')
  print("Real translation: this is the first book i've ever done.")

if __name__ == '__main__':
  train_new_transformer()