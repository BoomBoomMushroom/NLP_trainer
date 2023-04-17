import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

# Define the corpus
maxLen = 50

# ======                        Do it ourself bootleg                                              ======
import re
import regex
import nltk
def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

def cleanString(text=""):
    text = deEmojify(text)
    text = re.sub(r'[0-9]', '', text)
    text = text.replace("<", "").replace("@", "").replace(">", "").replace("*", "")
    text = re.sub(r'http\S+', '', text)
    text = regex.sub(r'[^\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]', '', text)
    return text

def readText(fileName="training_data.txt"):
    # Open the text file and read its contents
    with open(fileName, "r") as file:
        text = ""
        for line in file:
            splitted = line.strip().split(": ")
            if len(splitted) == 1: continue
            text += splitted[1] + "\n"
        #text = file.read()
    return text

text = cleanString(readText("training_data.txt")).split("\n")
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)
sequences = tokenizer.texts_to_sequences(text)
padded_sequences = pad_sequences(sequences, maxlen=maxLen)

# ======                                                                                            ======
"""
#print(text)
# Load and preprocess text data
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts([' '.join(sentence) for sentence in text])
sequences = tokenizer.texts_to_sequences([' '.join(sentence) for sentence in text])
padded_sequences = pad_sequences(sequences, maxlen=maxLen,truncating="pre")
"""
#Build neural network model
model = tf.keras.Sequential([
tf.keras.layers.Embedding(len(tokenizer.word_index) + 1, 64, input_length=maxLen-1),
tf.keras.layers.LSTM(64, return_sequences=True),
tf.keras.layers.Dense(len(tokenizer.word_index) + 1, activation="softmax")
])

#Compile model
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
optimizer="adam",
metrics=["accuracy"],
run_eagerly=True)

import os
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# model.load_weights(checkpoint_path)
model.save_weights(checkpoint_path.format(epoch=0))

latest = tf.train.latest_checkpoint(checkpoint_dir)
if latest and input("Load mode? y/n") == 'y':
    model.load_weights(latest)
else:
    #Train model
    X = padded_sequences[:, :-1]
    y = padded_sequences[:, 1:]

    model.fit(X, y, epochs=10, verbose=1, callbacks=[cp_callback])

    model.save_weights(checkpoint_path.format(epoch=0))

#Generate text using model
seed_text = input("Input: ")
for i in range(100):
    encoded = tokenizer.texts_to_sequences([seed_text])[0]
    padded = pad_sequences([encoded], maxlen=maxLen, truncating="pre")
    prediction = model.predict(padded)[0]
    predicted_word_index = tf.argmax(prediction).numpy()
    predicted_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            predicted_word = word
            break
    seed_text += " " + predicted_word

print(seed_text)