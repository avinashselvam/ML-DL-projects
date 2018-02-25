import tensorflow as tf
import pandas as pd
import numpy as np
import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# load and split toy data from sklearn

iris = load_iris()
iris_X, iris_y = iris.data, iris.target
np.append(iris_X, np.ones(len(iris_X)), axis=1)
iris_y = pd.get_dummies(iris_y).values
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3, random_state=2018)

# placeholders

no_features = X_train.shape[1]
no_labels = y_train.shape[1]

X = tf.placeholder(tf.float32, [None, no_features])
y = tf.placeholder(tf.float32, [None, no_labels])

W = tf.Variable(tf.random_normal([no_features, no_labels], mean=0, stddev=0.01, name='weights'))

# computational graph

mul_by_weights = tf.matmul(X, W, name='matmul')
activation = tf.nn.sigmoid(mul_by_weights, name='activation')

no_epochs = 500

learning_rate = tf.train.exponential_decay(learning_rate=0.0008, global_step=1, decay_steps=len(X_train), decay_rate=0.95, staircase=True)

loss = tf.nn.l2_loss(activation - y, name='squared_error_loss')

training = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

correct_predictions = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'))

cost = 0
diff = 1
epoch_values = []
accuracy_values = []
cost_values = []

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    for i in range(no_epochs):
        if i > 1 and diff < 0.0001:
            print('change in cost %g; convergence'%diff)
            break
        else:
            step = session.run(training, feed_dict={X: X_train, y: y_train})
            if i % 10 == 0:
                epoch_values.append(i)
                train_accuracy, newCost = session.run([accuracy, loss], feed_dict={X: X_train, y: y_train})
                accuracy_values.append(train_accuracy)
                cost_values.append(newCost)
                diff = abs(newCost - cost)
                cost = newCost

                print("step %d, training accuracy %g, cost %g, change in cost %g"%(i, train_accuracy, newCost, diff))

    print("final accuracy on test set: %s" %str(session.run(accuracy, 
                                                        feed_dict={X: X_test, 
                                                                    y: y_test})))
