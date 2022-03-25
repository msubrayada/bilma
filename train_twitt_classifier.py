from bilma import bilma_model

import numpy as np
import twitt_mask_seq_wordpiece_classification
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
import tensorflow as tf

from tensorflow.keras.layers import Dense, Dropout
import getopt, sys

arglist = sys.argv[1:]
options = "hc:e:t:m:"
long_options = ["Help", "country=", "epochs=" "trained_batches=", "model_epoch="]

country = "AR"
tr_b = 0
train_epochs = 0
model_epoch = 1
try:
    arguments, values = getopt.getopt(arglist, options, long_options)
    for currentArgument, currentValue in arguments:
        if currentArgument in ("-h", "--Help"):
            print ("Use: -c country -e epochs -t trained_batches")

        elif currentArgument in ("-c", "--country"):            
            country = currentValue
        elif currentArgument in ("-e", "--epochs"):            
            train_epochs = int(currentValue)
        elif currentArgument in ("-t", "--trained_batches"):                        
            tr_b = int(currentValue)
        elif currentArgument in ("-m", "--model_epoch"):
            model_epoch = int(currentValue)
        
            

except getopt.error as err:
    print (str(err))

path = "d:/data/twitts/"

twitt_file = f"{country}-emojis-train.txt"
vocab_file = "vocab_file_ALL.txt"
vocab_size = 29025
max_length = 280
batch_size = 64

twitt_gen = twitt_mask_seq_wordpiece_classification.seq_twitt(path + twitt_file, path + vocab_file, vocab_size, max_length=max_length, batch_size=batch_size, starting_batch=0)
l = len(twitt_gen)
v = int(len(twitt_gen)*0.8)
val_twitt_gen = twitt_mask_seq_wordpiece_classification.seq_twitt(path + twitt_file, path + vocab_file, vocab_size, max_length=max_length, batch_size=batch_size, starting_batch=v, training=False)





n = 2
embed_dim = 512
heads = 4

model = bilma_model.load(f"models/bert_small_{country}.txt_epoch-{model_epoch}_b-{tr_b}-final.h5")

x = tf.squeeze(model.layers[-2].output[:, 0:1, :], axis=1)
#x = Dropout(0.25)(x)
x = Dense(embed_dim, activation="relu", name="ex_1")(x)
x = Dense(embed_dim, activation="relu", name="ex_2")(x)
x = Dense(15, activation="softmax", name="cp")(x)
clf_model = Model(inputs=model.inputs, outputs=[model.outputs[0], x])

optimizer = tf.keras.optimizers.Adam(learning_rate = 0.000005, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False)
losses = [bilma_model.loss_function(), "categorical_crossentropy"]
loss_weights = [0.5, 0.5]
metrics = [None, "acc"]
clf_model.compile(optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)

callback = keras.callbacks.EarlyStopping(monitor='val_cp_acc', patience=2, mode="max", verbose=1)

history = clf_model.fit(twitt_gen, steps_per_epoch=v, shuffle=False, validation_data=val_twitt_gen, validation_steps=l-v, epochs=train_epochs,  verbose=1, workers=8, callbacks=[callback])

train_epochs = len(history.history['val_cp_acc'])
model_name = f"bert_small_{country}.txt_epoch-{model_epoch}_b-{tr_b}-final_classification_fcl-2_epochs-{train_epochs}.h5"
clf_model.save(f"models/{model_name}")

twitt_file = f"{country}-emojis-test.txt"
test_twitt_gen = twitt_mask_seq_wordpiece_classification.seq_twitt(path + twitt_file, path + vocab_file, vocab_size, max_length=max_length, batch_size=batch_size, starting_batch=0, training=False)


y_trues = []
y_preds = []

for x, y in test_twitt_gen:
    p = clf_model(x)
    y_preds.append(p[1])
    y_trues.append(y[1])
m = tf.keras.metrics.CategoricalAccuracy()
m.update_state(y_trues, y_preds)
res = m.result().numpy()
res_string = f"{country},{model_name},{res}"

with open(f"models/clf_results.txt", "a") as f:
    f.write(res_string + "\n")

print(res)

print("The end")
