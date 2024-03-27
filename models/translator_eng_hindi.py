import tensorflow as tf
import tensorflow_addons as tfa

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

from data.data_utils import hindi_characters_pattern
import unicodedata
import re
import numpy as np
import os
import io
import time
# http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip -> dataset of spa-eng
profiling_log_dir = "./profile_logs"
tf.profiler.experimental.start(logdir=profiling_log_dir)        # I am not able to run this, as for analyzing GPU, CUPTI is needed -> which could not be found for some reasons (cuptiSubscribe: error 15: CUPTI_ERROR_NOT_INITIALIZED) -> will try to upgrade once I remove dependency on tensorflow-addon (which is holding all teh versions down)
# disabling GPU 


class NMTDataset:
  def __init__(self, problem_type='eng-hindi',threshold=None):
      self.problem_type = problem_type
      self.inp_lang_tokenizer = None
      self.targ_lang_tokenizer = None
      self.threshold = threshold

  def unicode_to_ascii(self, s):
      return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

  ## Step 1 and Step 2 
  def preprocess_sentence_hindi(self, w):
      # Create a space between a word and the punctuation following it
      # eg: "वह लड़का है." => "वह लड़का है ."
      # Reference: https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
      # Replace everything with space except (a-z, A-Z, ".", "?", "!", ",", "।", Hindi characters)
      # Reference: https://stackoverflow.com/questions/76658990/regex-expression-to-validate-only-hindi-devnagri-letters-in-python
      w = re.sub(r"[^?.!,"+hindi_characters_pattern+"]", " ", w)        # discarded english terms also, if needed (Ideally should not be given)
      w = re.sub(r"([?.!,।])", r" \1 ", w)
      w = re.sub(r'[" "]+', " ", w)
      w = w.strip()

      # Add a start and an end token to the sentence
      # so that the model knows when to start and stop predicting.
      w = '<start> ' + w + ' <end>'
      return w
  def preprocess_sentence_english(self, w):
      w = self.unicode_to_ascii(w.lower().strip())

      # creating a space between a word and the punctuation following it
      # eg: "he is a boy." => "he is a boy ."
      # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
      w = re.sub(r"([?.!,¿])", r" \1 ", w)
      w = re.sub(r'[" "]+', " ", w)

      # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
      w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
      w = w.strip()

      # adding a start and an end token to the sentence
      # so that the model know when to start and stop predicting.
      w = '<start> ' + w + ' <end>'
      return w
  
  def create_dataset(self, path, num_examples):
      # path : path to spa-eng.txt file
      # num_examples : Limit the total number of training example for faster training (set num_examples = len(lines) to use full data)
      lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
      # lines contains each of the lines as it is in .txt ... like 0->'Go.\tVe.', 1->'Go.\tVete.'...
      word_pairs = []
      print("WordPairing started")
      for l in lines[:num_examples]:
          splited = l.split('\t')
          hindi = self.preprocess_sentence_hindi(splited[1])
          english = self.preprocess_sentence_english(splited[0])
          if (not self.threshold) or (len(hindi.split(' '))<=self.threshold+2  and len(english.split(' '))<=self.threshold+2 + 10):       # this extra 10 is some extra space for input
              word_pairs.append([english, hindi])
      # word_pairs = [['<start> go . <end>', '<start> ve . <end>'],['<start> go . <end>', '<start> vete . <end>'],...]
      print("WordPairs returned")
      return zip(*word_pairs)

  # Step 3 and Step 4
  def tokenize(self, lang):
      # lang = list of sentences in a language
      
      print(len(lang), "example sentence: {}".format(lang[-1]))
      print(len(lang), "example sentence: {}".format(lang[4]))
      lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<OOV>')
      lang_tokenizer.fit_on_texts(lang)

      ## tf.keras.preprocessing.text.Tokenizer.texts_to_sequences converts string (w1, w2, w3, ......, wn) 
      ## to a list of correspoding integer ids of words (id_w1, id_w2, id_w3, ...., id_wn)
      tensor = lang_tokenizer.texts_to_sequences(lang) 

      ## tf.keras.preprocessing.sequence.pad_sequences takes argument a list of integer id sequences 
      ## and pads the sequences to match the longest sequences in the given input
      tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

      return tensor, lang_tokenizer

  def load_dataset(self, path, num_examples=None):
      # creating cleaned input, output pairs
      # targ_lang, inp_lang = self.create_dataset(path, num_examples)             # changing the target and input if necessary
      inp_lang, targ_lang  = self.create_dataset(path, num_examples)
      # targ_lang = ('<start> go . <end>','<start> go . <end>',...)
      # inp_lang = ('<start> ve . <end>','<start> vete . <end>',...)
      input_tensor, inp_lang_tokenizer = self.tokenize(inp_lang)
      target_tensor, targ_lang_tokenizer = self.tokenize(targ_lang)
      # target_tensor = [array([ 2, 37,  4,  3,  0,  0,  0,  0,  0,  0,  0]),array([ 2, 37,  4,  3,  0,  0,  0,  0,  0,  0,  0]),...]
      # input_tensor = [array([ 2, 136,  4,  3,  0,  0,  0,  0,  0,  0,  0]),array([ 2, 294,  4,  3,  0,  0,  0,  0,  0,  0,  0]),...]
      return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

  def call(self, num_examples, BUFFER_SIZE, BATCH_SIZE):
      file_path = "data/translation/en-hi.txt"
      input_tensor, target_tensor, self.inp_lang_tokenizer, self.targ_lang_tokenizer = self.load_dataset(file_path, num_examples)
      # target_tensor = [array([ 2, 37,  4,  3,  0,  0,  0,  0,  0,  0,  0]),array([ 2, 37,  4,  3,  0,  0,  0,  0,  0,  0,  0]),...]
      # input_tensor = [array([ 2, 136,  4,  3,  0,  0,  0,  0,  0,  0,  0]),array([ 2, 294,  4,  3,  0,  0,  0,  0,  0,  0,  0]),...]
      input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

      train_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)) # -> contains list of 64 trainingXY tensors -> each batch's each tensor is pair of tensors -> both containing tokenized words
      train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

      val_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_val, target_tensor_val))
      val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)

      return train_dataset, val_dataset, self.inp_lang_tokenizer, self.targ_lang_tokenizer

num_examples = 1659084//10                    # 1659084 <- total data size (25% taken)
BUFFER_SIZE = num_examples + 1                     # Buffer size > number of samples
BATCH_SIZE = 256
# Let's limit the #training examples for faster training
profiling_log_dir = "./profile_logs"
# tf.profiler.experimental.start(logdir=profiling_log_dir)
dataset_creator = NMTDataset('eng-hindi',threshold = 36)
train_dataset, val_dataset, inp_lang, targ_lang = dataset_creator.call(num_examples, BUFFER_SIZE, BATCH_SIZE)
# tf.profiler.experimental.stop(save=True)
example_input_batch, example_target_batch = next(iter(train_dataset))
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1
max_length_input = example_input_batch.shape[1]
max_length_output = example_target_batch.shape[1]

embedding_dim = 256
units = 1024
steps_per_epoch = num_examples//BATCH_SIZE

##### 

class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)                  # input embedding -> vocab_size * embedding_dim -> Network's lookup table (https://www.tensorflow.org/text/guide/word_embeddings) -> also knows the relationship between words

    ##-------- LSTM layer in Encoder ------- ##
    self.lstm_layer = tf.keras.layers.LSTM(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    


  def call(self, x, hidden):
    x = self.embedding(x)
    print("Encoding: ",x)
    output, h, c = self.lstm_layer(x, initial_state = hidden)
    return output, h, c

  def initialize_hidden_state(self):
    return [tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units))] 


encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, attention_type='luong'):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.attention_type = attention_type
    
    # Embedding Layer
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    
    #Final Dense layer on which softmax will be applied
    self.fc = tf.keras.layers.Dense(vocab_size)

    # Define the fundamental cell for decoder recurrent structure
    self.decoder_rnn_cell = tf.keras.layers.LSTMCell(self.dec_units)
   


    # Sampler
    self.sampler = tfa.seq2seq.sampler.TrainingSampler()

    # Create attention mechanism with memory = None
    self.attention_mechanism = self.build_attention_mechanism(self.dec_units, 
                                                              None, self.batch_sz*[max_length_input], self.attention_type)

    # Wrap attention mechanism with the fundamental rnn cell of decoder
    self.rnn_cell = self.build_rnn_cell(batch_sz)

    # Define the decoder with respect to fundamental rnn cell
    self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell, sampler=self.sampler, output_layer=self.fc)

    
  def build_rnn_cell(self, batch_sz):
    rnn_cell = tfa.seq2seq.AttentionWrapper(self.decoder_rnn_cell, 
                                  self.attention_mechanism, attention_layer_size=self.dec_units)
    return rnn_cell

  def build_attention_mechanism(self, dec_units, memory, memory_sequence_length, attention_type='luong'):
    # ------------- #
    # typ: Which sort of attention (Bahdanau, Luong)
    # dec_units: final dimension of attention outputs 
    # memory: encoder hidden states of shape (batch_size, max_length_input, enc_units)
    # memory_sequence_length: 1d array of shape (batch_size) with every element set to max_length_input (for masking purpose)

    if(attention_type=='bahdanau'):
      return tfa.seq2seq.BahdanauAttention(units=dec_units, memory=memory, memory_sequence_length=memory_sequence_length)
    else:
      return tfa.seq2seq.LuongAttention(units=dec_units, memory=memory, memory_sequence_length=memory_sequence_length)

  def build_initial_state(self, batch_sz, encoder_state, Dtype):
    decoder_initial_state = self.rnn_cell.get_initial_state(batch_size=batch_sz, dtype=Dtype)
    decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
    return decoder_initial_state


  def call(self, inputs, initial_state):
    x = self.embedding(inputs)
    outputs, _, _ = self.decoder(x, initial_state=initial_state, sequence_length=self.batch_sz*[max_length_output-1])
    return outputs

decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE, 'luong')

optimizer = tf.keras.optimizers.Adam()

def loss_function(real, pred):
  # real shape = (BATCH_SIZE, max_length_output)
  # pred shape = (BATCH_SIZE, max_length_output, tar_vocab_size )
  cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
  loss = cross_entropy(y_true=real, y_pred=pred)
  mask = tf.logical_not(tf.math.equal(real,0))   #output 0 for y=0 else output 1
  mask = tf.cast(mask, dtype=loss.dtype)  
  loss = mask* loss
  loss = tf.reduce_mean(loss)
  return loss  

checkpoint_dir = './training_checkpoints_ENG_HINDI'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckptEngHindi")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

@tf.function
def train_step(inp, targ, enc_hidden):
  loss = 0

  with tf.GradientTape() as tape:
    enc_output, enc_h, enc_c = encoder(inp, enc_hidden)


    dec_input = targ[ : , :-1 ] # Ignore <end> token
    real = targ[ : , 1: ]         # ignore <start> token

    # Set the AttentionMechanism object with encoder_outputs
    decoder.attention_mechanism.setup_memory(enc_output)

    # Create AttentionWrapperState as initial_state for decoder
    decoder_initial_state = decoder.build_initial_state(BATCH_SIZE, [enc_h, enc_c], tf.float32)
    pred = decoder(dec_input, decoder_initial_state)
    logits = pred.rnn_output
    loss = loss_function(real, logits)

  variables = encoder.trainable_variables + decoder.trainable_variables
  gradients = tape.gradient(loss, variables)
  optimizer.apply_gradients(zip(gradients, variables))

  return loss
gpu_device = '/gpu:0'
EPOCHS = 10
try:
  with tf.device(gpu_device):
    for epoch in range(EPOCHS):
        start = time.time()
        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0
        for (batch, (inp, targ)) in enumerate(train_dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ, enc_hidden)
            total_loss += batch_loss
            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                        batch,
                                                        batch_loss.numpy()))
        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

except Exception as e:
    print("An error occurred during training:", e)    

tf.profiler.experimental.stop(save=True)
