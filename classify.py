import os
import numpy
import matplotlib.image
import tensorflow
from tensorflow.contrib.slim.python.slim.model_analyzer import tensor_description

path = os.path.join('mlsp_contest_dataset', 'supplemental_data', 'spectrograms')

learning_steps = 4096

input_size_height = 256
input_size_width = 1246
output_classes = 19

def weight_variable(shape):
    initial = tensorflow.truncated_normal(shape, stddev=0.1)
    return tensorflow.Variable(initial)

def bias_variable(shape):
    initial = tensorflow.constant(0.1, shape=shape)
    return tensorflow.Variable(initial)

def prepare_dataset():

    labels = []

    file_labels = open(os.path.join('mlsp_contest_dataset', 'essential_data', 'rec_labels_test_hidden.txt'))
    file_labels.readline()

    images = []
    for file in os.listdir(path):

        file_path = os.path.abspath(os.path.join(path, file))
        _, extension = os.path.splitext(file_path)

        if extension != '.bmp':
            continue

        line = file_labels.readline()
        if line.find('?') > 0:
            continue

        classes = line.split(',')
        label = numpy.zeros((1, output_classes))

        for c in range(1, len(classes)):
            label[(0,classes[c])] = 1.0

        image = matplotlib.image.imread(file_path)

        images.append(image)
        labels.append(label)

    return (images, labels)

def conv2d(x, W):
    return tensorflow.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tensorflow.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x_place = tensorflow.placeholder(tensorflow.float32, shape=[None, input_size_height * input_size_width])
y_place = tensorflow.placeholder(tensorflow.float32, shape=[None, output_classes])

x_image = tensorflow.reshape(x_place, [-1, input_size_height, input_size_width, 1])

# Create 1st convolution layer
W_conv_1st = weight_variable([5, 5, 1, 32])
b_conv_1st = bias_variable([32])


h_conv_1st = tensorflow.nn.relu(conv2d(x_image, W_conv_1st) + b_conv_1st)
h_pool_1st = max_pool_2x2(h_conv_1st)


# Create 2nd convolution layer
W_conv_2nd = weight_variable([5, 5, 32, 64])
b_conv_2nd = bias_variable([64])

h_conv_2nd = tensorflow.nn.relu(conv2d(h_pool_1st, W_conv_2nd) + b_conv_2nd)
h_pool_2nd = max_pool_2x2(h_conv_2nd)

# Fully connected 1st layer
W_fc_1st = weight_variable([7 * 7 * 64, 1024])
b_fc_1st = bias_variable([1024])

h_pool_2nd_flat = tensorflow.reshape(h_pool_2nd, [-1, 7 * 7 * 64])
h_fc_1st = tensorflow.nn.relu(tensorflow.matmul(h_pool_2nd_flat, W_fc_1st) + b_fc_1st)

# Fully connected 2nd layer
W_fc_2nd = weight_variable([1024, output_classes])
b_fc_2nd = bias_variable([output_classes])

y_conv = tensorflow.matmul(h_fc_1st, W_fc_2nd) + b_fc_2nd

cross_entropy = tensorflow.reduce_mean(tensorflow.nn.softmax_cross_entropy_with_logits(y_conv, y_place))
train_step = tensorflow.train.AdamOptimizer(0.005).minimize(cross_entropy)

correct_prediction = tensorflow.equal(tensorflow.argmax(y_conv, 1), tensorflow.argmax(y_place, 1))
accuracy = tensorflow.reduce_mean(tensorflow.cast(correct_prediction, tensorflow.float32))

session = tensorflow.InteractiveSession()
session.run(tensorflow.global_variables_initializer())

(images, labels) = prepare_dataset()
batch_size = 1
index = 0

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
t = mnist.train.next_batch(50)

for i in range(learning_steps):

    batch = numpy.ndarray((batch_size, input_size_height, input_size_width))
    label = numpy.ndarray((batch_size, output_classes))

    for j in range(batch_size):

        batch[j] = images[index % len(images)]
        label[j] = labels[index % len(images)]
        ++index

    if i % 64 == 0:
        train_accuracy = accuracy.eval(feed_dict={x_place: batch, y_place: label})
        print("Step %d: training accuracy = %g" % (i, train_accuracy))

    train_step.run(feed_dict={x_place: batch[0], y_place: batch[1]})

