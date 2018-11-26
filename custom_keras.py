import tensorflow as tf 

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation = tf.nn.relu),
    tf.keras.layers.Dense(10, activation = tf.nn.relu),
    tf.keras.layers.Dense(3)
])

