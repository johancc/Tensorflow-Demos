import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from math import pi 
import tensorflow as tf 
import numpy as np
tf.enable_eager_execution()
tfe = tf.contrib.eager #shortcut



def f(x):
    return tf.square(tf.sin(x))

assert f(pi/2).numpy() == 1.0

grad_f = tfe.gradients_function(f)

# the derivative of the function with its arguments is calculated

assert tf.abs(grad_f(pi/2)[0]).numpy() < 1e-7

def grad(f):
  return lambda x: tfe.gradients_function(f)(x)[0]

x = tf.lin_space(-2*pi, 2*pi, 100)  # 100 points between -2π and +2π

import matplotlib.pyplot as plt

plt.plot(x, f(x), label="f")
plt.plot(x, grad(f)(x), label="first derivative")
plt.plot(x, grad(grad(f))(x), label="second derivative")
plt.plot(x, grad(grad(grad(f)))(x), label="third derivative")
plt.legend()
# plt.show()

x = tf.ones((2,2))

with tf.GradientTape(persistent=True) as tape:
    tape.watch(x)
    y = tf.reduce_sum(x)
    z = tf.multiply(y,y)

# compute the derivative for an intermediate value
dz_dy = tape.gradient(z,y)
assert dz_dy.numpy() == 8.0

dz_dx = tape.gradient(z,x)
for i in [0,1]:
    for j in [0,1]:
        assert dz_dx[i][j].numpy() == 8.0


# using multiple Gradient Tapes
x = tf.constant(1.0)  # Convert the Python 1.0 to a Tensor object

with tf.GradientTape() as t:
  with tf.GradientTape() as t2:
    t2.watch(x)
    y = x * x * x
  # Compute the gradient inside the 't' context manager
  # which means the gradient computation is differentiable as well.
  dy_dx = t2.gradient(y, x)
d2y_dx2 = t.gradient(dy_dx, x)

assert dy_dx.numpy() == 3.0
assert d2y_dx2.numpy() == 6.0
