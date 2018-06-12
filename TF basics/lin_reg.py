import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# make data

x = np.random.rand(100)
y = 5*x + 1
y = np.vectorize(lambda y: y + np.random.normal(0,0.1))(y)

# get idea about data

print('Example data')
print('------------')
print(list(zip(x,y))[0:5])

# computational graph

w0 = tf.Variable(0.0)
w1 = tf.Variable(0.0)
y_ = w1*x + w0

# gradient learning on loss function

loss = tf.reduce_mean(tf.square(y - y_))

learning_rate = 0.5
optimiser = tf.train.GradientDescentOptimizer(learning_rate)
train = optimiser.minimize(loss)

# init and run graph

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    for iter in range(500):
        evals = session.run([train, w0, w1])[1:]
        if iter % 20 == 0:
            print(iter, evals)
            

