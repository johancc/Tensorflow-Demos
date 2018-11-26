import tensorflow as tf 
import matplotlib.pyplot as plt
tf.enable_eager_execution()

class Model(object):
    def __init__(self):
        # initialize weight and bias
        self.W = tf.Variable(5.0)
        self.b = tf.Variable(0.0)
    def __call__(self, x):
        return self.W * x + self.b

def loss(predicted, desired):
    return tf.reduce_mean(tf.square(predicted - desired))

model = Model()
TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000

inputs  = tf.random_normal(shape=[NUM_EXAMPLES])
noise   = tf.random_normal(shape=[NUM_EXAMPLES])
outputs = inputs * TRUE_W + TRUE_b + noise

def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(model(inputs), outputs)
    dW, db = t.gradient(current_loss, [model.W, model.b])
    model.W.assign_sub(learning_rate * dW)
    model.b.assign_sub(learning_rate * db)

model = Model()
# looping.
weights, biases = [],[]
epochs = range(10)
for epoch in epochs:
    weights.append(model.W.numpy())
    biases.append(model.b.numpy())
    current_loss = loss(model(inputs), outputs)

    train(model, inputs, outputs, learning_rate = 0.1)
    print('Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f' %
        (epoch, weights[-1], biases[-1], current_loss))
    
# history
plt.plot(epochs, weights, 'r',
         epochs, biases, 'b')
plt.plot([TRUE_W] * len(epochs), 'r--',
         [TRUE_b] * len(epochs), 'b--')
plt.legend(['W', 'b', 'true W', 'true_b'])
plt.show()