import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
from random import randint

# data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# set up placeholder. 784 is the size of the image matrix 28x28 pixels
image_x = 28
image_y = 28
img_size = image_x * image_y
x = tf.placeholder(tf.float32, [None, img_size])

# 10 digit classifications
num_types = 10
W = tf.Variable(tf.zeros([img_size, num_types]))
b = tf.Variable(tf.zeros([num_types]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, num_types])

cross_entropy = -tf.reduce_sum(y_* tf.log(y) )

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()

sess.run(init)

# train
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))


accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print "Accuracy: ",(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

# test a random image from the sample data
num = randint(0, mnist.test.images.shape[0])
rand_img = mnist.test.images[num]

prediction = sess.run(tf.argmax(y,1), feed_dict={x:[rand_img]})
plt.imshow(rand_img.reshape(28, 28), cmap=plt.cm.binary)
print 'Prediction: ', prediction
plt.show()
