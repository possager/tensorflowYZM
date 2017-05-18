import tensorflow as tf
from captcha.image import ImageCaptcha
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random
import string






characters = string.digits + string.ascii_uppercase+string.ascii_lowercase
print(characters)
width, height, n_len, n_class = 210, 100, 5, len(characters)




def mnistTest():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    x = tf.placeholder("float", [None, 784])
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x,W) + b)
    y_ = tf.placeholder("float", [None,10])
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for i in range(1000):
      batch_xs, batch_ys = mnist.train.next_batch(100)
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})



def gen(batch_size=32):
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]
    generator = ImageCaptcha(width=width, height=height)
    while batch_size>0:
        for i in range(batch_size):
            random_str = ''.join([random.choice(characters) for j in range(5)])
            print random_str
            X[i] = generator.generate_image(random_str)
            plt.show(X[i])
            for j, ch in enumerate(random_str):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1
        yield X, y
        batch_size-=1


def decode(y):
    y = np.argmax(np.array(y), axis=2)[:,0]
    return ''.join([characters[x] for x in y])

X, y = next(gen(1))
# plt.imshow(X[0])
img1=Image.fromarray(X[0])
img1.show()
plt.title(decode(y))