from __future__ import absolute_import, division, print_function
from tensorflow import keras 

import tensorflow as tf 
import numpy as np
import os

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images[:1000].reshape(-1, 28 ** 2)/255.0
test_images = test_images[:1000].reshape(-1, 28 ** 2)/255.0

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation = tf.nn.relu, input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation = tf.nn.softmax)
    ])
    model.compile(optimizer = tf.keras.optimizers.Adam(), 
                  loss = tf.keras.losses.sparse_categorical_crossentropy,
                  metrics = ['accuracy'])
    return model

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


model = create_model()
# model.fit(train_images, train_labels, epochs = 10, 
#          validation_data= (test_images, test_labels), 
#          callbacks = [cp_callback]) # inject the callback

loss, acc  = model.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))
model.load_weights(checkpoint_path)
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

# restoring models every n epochs
# include the epoch in the file name. (uses `str.format`)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # every 5-epochs.
    period=5)

model = create_model()
# model.fit(train_images, train_labels,
#           epochs = 50, callbacks = [cp_callback],
#           validation_data = (test_images,test_labels),
#           verbose=0)

# using the last checkpoint
latest = tf.train.latest_checkpoint(checkpoint_dir)
model = create_model()
model.load_weights(latest)
loss, acc = model.evaluate(test_images, test_labels)

print("Restored model from latest, accuracy: {:5.2f}%".format(100 * acc))

def manual_save(model, path = './checkpoints/my_checkpoint'):
    model.save_weights(path)
    model = create_model()
    model.load_weights(path)

    _, acc = model.evaluate(test_images, test_labels)
    print("Manually saved model, accuracy: {:5.2f}%".format(acc * 100))

manual_save(model)

# saying the full model
model = create_model()
# model.fit(train_images, train_labels, epochs = 5)

# try:
#     model.save("./models/my_model.h5")
# except OSError:
#     # folder not there
#     os.mkdir("/models")
#     model.save("./models/my_model.h5")

# verifying
new_model = keras.models.load_model("./models/my_model.h5")
new_model.summary()
_, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}$")