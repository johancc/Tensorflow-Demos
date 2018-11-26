import tensorflow as tf 
from tensorflow import keras
import numpy as np 

imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000)

# data is preprocessed
# print("training: {}, labels: {}".format(len(train_data), len(train_labels)))

# decoding logic

word_index = imdb.get_word_index()

# making room
word_index = {k: (v + 3) for k,v in word_index.items()}
# inserting base cases
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reversed_word_index = dict([(value,key) for (key, value) in word_index.items()])
def decode_review(text):
    return " ".join([reversed_word_index.get(i, "?") for i in text])

# print(decode_review(train_data[0]))

# the reviews are of different sizes, so the dataset needs to be padded.

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

# this is a binary classifier
vocab_size = 10000
model = keras.Sequential()
# integer-encoded  vocab -> embedding vector for each index
model.add(keras.layers.Embedding(vocab_size, 16))

# averages over the sequence dimension.
model.add(keras.layers.GlobalAveragePooling1D())

# 16 hidden units
model.add(keras.layers.Dense(16, activation=tf.nn.relu))

# output will be the probability.
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

# model.summary()

model.compile(optimizer = tf.train.AdamOptimizer(), 
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])                                     
                                
# validation sets
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(partial_x_train, 
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose = 1)

results = model.evaluate(test_data, test_labels)
print(results)


def training_vs_validation():
    """ displays training vs validation over the epochs """
    import matplotlib.pyplot as plt

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo" = "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b = "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# training_vs_validation() 
# shows that there is overfitting 
