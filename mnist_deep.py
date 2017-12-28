# importing tensorflow and MNIST data
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import math

# reading and storing data into a python variable.
# one_hot encoding is vectorising categorical data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# for reproducible results
tf.set_random_seed(0)

# placeholder to recieve and hold each entry from data
X = tf.placeholder(tf.float32, [None,28,28,1])

# reshape into linear vector to simplify network
X = tf.reshape(X, [-1, 784])

# create and initialize weights and basis matrices
# 5 layer network
W1 = tf.Variable(tf.truncated_normal([784,200],stddev=0.1))
b1 = tf.Variable(tf.ones([200]) / 10)

W2 = tf.Variable(tf.truncated_normal([200, 100],stddev=0.1))
b2 = tf.Variable(tf.ones([100]) / 10)

W3 = tf.Variable(tf.truncated_normal([100, 60], stddev=0.1))
b3 = tf.Variable(tf.ones([60]) / 10)

W4 = tf.Variable(tf.truncated_normal([60, 30], stddev=0.1))
b4 = tf.Variable(tf.ones([30]) / 10)

W5 = tf.Variable(tf.truncated_normal([30, 10], stddev=0.1))
b5 = tf.Variable(tf.ones([10]) / 10)

# tensorflow requires variables to be initialised before execution
init = tf.global_variables_initializer()

# the actual network model
Y1 = tf.nn.relu(tf.matmul(X, W1) + b1)
Y2 = tf.nn.relu(tf.matmul(Y1, W2) + b2)
Y3 = tf.nn.relu(tf.matmul(Y2, W3) + b3)
Y4 = tf.nn.relu(tf.matmul(Y3, W4) + b4)
Ylogits = tf.matmul(Y4, W5) + b5
Y  = tf.nn.softmax(Ylogits)

# placeholder to recieve and hold each label from data
Y_ = tf.placeholder(tf.float32,[None,10])

# loss function, -ve if correctly classified, 0 otherwise
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# checks if index of max porbability is same as one_hot label
is_correct = tf.equal(tf.argmax(Y,1),tf.argmax(Y_,1))

# correctly classified by total input quantity
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

# parameter for gradient descent, a big value will cause bouncing
#learning_rate = 0.003
lr = tf.placeholder(tf.float32)

lrmin = 0.0001
lrmax = 0.003


# Define the training process with gradient descent optimizer with rate and loss function
optimizer = tf.train.GradientDescentOptimizer(lr)
train_step = optimizer.minimize(cross_entropy)

# initialise session and execute training steps
sess = tf.Session()
sess.run(init)

for i in range(5000):

    # input to be fed as a dictionary
    batch_X, batch_Y = mnist.train.next_batch(100)
    learning_rate = lrmin + (lrmax - lrmin) * math.exp(-i / 2000)

    train_data = {X: batch_X, Y_: batch_Y, lr: learning_rate}

    sess.run(train_step, feed_dict = train_data)

    train_acc, train_entropy = sess.run([accuracy, cross_entropy], feed_dict=train_data)
    print('At iteration {} model has training accuracy of {}'.format(i+1,train_acc))

# determine test accuracy by changing input feed
test_data = {X:mnist.test.images, Y_: mnist.test.labels}
test_acc, test_entropy = sess.run([accuracy,cross_entropy],feed_dict=test_data)

print('Model has test accuracy of {}'.format(test_acc))
