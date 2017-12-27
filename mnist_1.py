# importing tensorflow and MNIST data
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# reading and storing data into a python variable.
# one_hot encoding is vectorising categorical data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# placeholder to recieve and hold each entry from data
X = tf.placeholder(tf.float32, [None,28,28,1])

# reshape into linear vector to simplify network
X = tf.reshape(X, [-1, 784])

# create and initialize weights and basis matrix
# 10x784(28x28) one set of weights, one basis for each of 10 output neurons
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# tensorflow requires variables to be initialised before execution
init = tf.initialize_all_variables()

# the actual network model
Y = tf.nn.softmax(tf.matmul(X,W)+b)

# placeholder to recieve and hold each label from data
Y_ = tf.placeholder(tf.float32,[None,10])

# loss function, -ve if correctly classified, 0 otherwise
cross_entropy = -tf.reduce_sum(Y_*tf.log(Y))

# checks if index of max porbability is same as one_hot label
is_correct = tf.equal(tf.argmax(Y,1),tf.argmax(Y_,1))

# correctly classified by total input quantity
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

# parameter for gradient descent, a big value will cause bouncing
learning_rate = 0.003

# Define the training process with gradient descent optimizer with rate and loss function
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_step = optimizer.minimize(cross_entropy)

# initialise session and execute training steps
sess = tf.Session()
sess.run(init)

for i in range(1000):

    # input to be fed as a dictionary
    batch_X, batch_Y = mnist.train.next_batch(100)
    train_data = {X: batch_X, Y_: batch_Y}

    sess.run(train_step, feed_dict = train_data)

    train_acc, train_entropy = sess.run([accuracy, cross_entropy], feed_dict=train_data)
    print('At iteration {} model has training accuracy of {}'.format(i+1,train_acc))

# determine test accuracy by changing input feed
test_data = {X:mnist.test.images, Y_: mnist.test.labels}
test_acc, test_entropy = sess.run([accuracy,cross_entropy],feed_dict=test_data)

print('Model has test accuracy of {}'.format(test_acc))
