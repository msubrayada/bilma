import bilma.transformer_text as transformer
from tensorflow.keras.layers import  Input,  Dense, Embedding

from tensorflow.keras.models import Model, load_model
import tensorflow as tf

import bilma.wordpiece_tokenizer

def loss_function(ignore_id=0):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    def loss(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, ignore_id))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        sum_ = tf.reduce_sum(mask,axis=1)
        
        loss_ = tf.math.divide_no_nan(tf.reduce_sum(loss_, axis=1), sum_)
        return loss_
    return loss

def accuracy_function(ignore_id=0):
    def acc_mlm(real, pred):
        accuracies = tf.equal(tf.cast(real, tf.int64), tf.argmax(pred, axis=2))

        mask = tf.math.logical_not(tf.math.equal(real, ignore_id))
        accuracies = tf.math.logical_and(mask, accuracies)

        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.math.divide_no_nan(tf.reduce_sum(accuracies), tf.reduce_sum(mask))
    return acc_mlm

def bilma(num_enc=6, embed_dim=300, max_length=50, num_heads=6, ff_dim=512, vocab_size=9739, rate=0.1):
    capt_inputs_ids = Input(shape=(max_length, ), name='capt_input')
    capt_embedding = Embedding(vocab_size, embed_dim, mask_zero=False, name="embedding")
    capt_inputs = capt_embedding(capt_inputs_ids)
    
    enc = transformer.Encoder(num_enc, embed_dim, max_length, num_heads, ff_dim, rate=rate)
    enc_output = enc(capt_inputs)
    fin_output = Dense(vocab_size, use_bias=True)(enc_output)
    
    caption_model = Model(inputs=capt_inputs_ids, outputs=[fin_output])
    return caption_model

def load(model_file):
    custom_objects={"EncoderBlock": transformer.EncoderBlock, 
                    "Encoder": transformer.Encoder,
                    "loss": loss_function(),
                    "acc_mlm":accuracy_function(),
                   }
    return load_model(model_file, custom_objects=custom_objects)

class tokenizer():
    def __init__(self, vocab_file, max_length):
        self.tokenizer = wordpiece_tokenizer.Tokenizer(vocab_file)
        self.max_length = max_length
        self.START = 2
        self.END = 3
        self.PAD = 0
        self.MASK = 4  
        
    def tokenize(self, text):
        c = self.tokenizer.tokenize([text])                       
        c = tf.concat([[[self.START]], c, [[self.END]]], axis=1)
        l_c = len(c[0])
        c = tf.pad(c, [[0,0],[0,self.max_length-l_c]])
        return c.numpy()
    
    def detokenize(self, tokens):
        words = self.tokenizer.detokenize(tokens)
        return tf.strings.reduce_join(words, separator=' ', axis=-1)
        
    
    