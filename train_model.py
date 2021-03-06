import os
import sys
import numpy
import tensorflow
from skimage.measure.tests.test_pnpoly import test_npnpoly
from tensorflow.contrib.slim.python.slim.model_analyzer import tensor_description

import matplotlib.pyplot as plot
import matplotlib

import prepare_data
import nn_schema

batch_size = 11
learning_epochs = 20
output_classes = 4
learning_rate = 0.000004

summaries_directory = (os.path.join(os.getcwd(), 'summary'))

input_size_height = prepare_data.image_rows
input_size_width = prepare_data.image_columns

keep_probability = tensorflow.placeholder(tensorflow.float32)

x_place = tensorflow.placeholder(tensorflow.float32, shape=[None, input_size_height, input_size_width])
print('Input tensor size = {}'.format(x_place.get_shape()))

y_place = tensorflow.placeholder(tensorflow.float32, shape=[None, output_classes])
print('Output tensor size = {}'.format(y_place.get_shape()))

x_image = tensorflow.reshape(x_place, [-1, input_size_height, input_size_width, 1])
print('Image tensor size = {}'.format(x_image.get_shape()))

with tensorflow.name_scope('output'):
    with tensorflow.name_scope('raw'):
        y_output = nn_schema.create_schema(x_image, output_classes, keep_probability)
    with tensorflow.name_scope('acivation'):
        y_output_softmax = tensorflow.nn.softmax(y_output)


with tensorflow.name_scope('cross_entropy'):
    with tensorflow.name_scope('difference'):
        softmax_cross_entropy_with_logits = tensorflow.nn.softmax_cross_entropy_with_logits(logits=y_output, labels=y_place)
        tensorflow.summary.histogram('sigmoid_cross_entropy_with_logits', softmax_cross_entropy_with_logits)

    with tensorflow.name_scope('total'):
        cross_entropy = tensorflow.reduce_mean(softmax_cross_entropy_with_logits)

tensorflow.summary.scalar('cross_entropy', cross_entropy)

with tensorflow.name_scope('train'):
    train_step = tensorflow.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

session = tensorflow.InteractiveSession()

merged = tensorflow.summary.merge_all()
train_writer = tensorflow.summary.FileWriter(os.path.join(summaries_directory, 'train'), session.graph)
test_writer = tensorflow.summary.FileWriter(os.path.join(summaries_directory, 'test'))

tensorflow.global_variables_initializer().run()

index = 0

(images, labels) = prepare_data.prepare_train_dataset()
print('Train data preparing is complete.')

print('Input image height = {} width = {}'.format(input_size_height, input_size_width))
print('Output classes = {}'.format(output_classes))

print('Batch size = {}'.format(batch_size))
print('Learning rate = {}'.format(learning_rate))

print('Train data size = {}'.format(len(images)))
print('Batch size = {}'.format(batch_size))
print('Learning epochs = {}'.format(learning_epochs))


batch = numpy.ndarray((batch_size, input_size_height, input_size_width))
label = numpy.ndarray((batch_size, output_classes))

step = 0
while(True):

    step += 1

    epoch = index / len(images) + 1
    if epoch >= learning_epochs:
        break

    for j in range(batch_size):

        batch[j, :] = images[index % len(images)]
        label[j, :] = labels[index % len(images)]

        index += 1

    summary, _, cross_entropy_batch = session.run([merged, train_step, cross_entropy], feed_dict = {x_place: batch, y_place: label, keep_probability: 0.5})

    print("Step #%d Epoch #%d: cross-entropy = %g, images = %d" % (step, epoch, cross_entropy_batch, index))
    test_writer.add_summary(summary, step)

print('Training is completed.\n')

test_data = False
if test_data:
    (test_files, test_images) = prepare_data.prepare_test_dataset()
    print('Test data preparing is complete.')
    print('Input image height = {} width = {}'.format(input_size_height, input_size_width))
    print('Output classes = {}'.format(output_classes))
    print('Test data size = {}'.format(len(test_images)))
    print('Starting evaluating predictions.\n')

    with open('test_predictions.csv', 'w') as test_predictions_file:

        test_predictions_file.write('clip,species,probability\n')

        for i in range(len(test_files)):
            file_name = test_files[i]
            test_image = test_images[i]

            output = numpy.zeros((1, output_classes))

            _, max_length = test_image.shape

            length = 0
            while length + prepare_data.image_columns < max_length:
                sample = test_image[:, length : length + prepare_data.image_columns]

                length += int(prepare_data.image_columns / 2)
                output_temp = y_output_softmax.eval(feed_dict = {x_place: sample.reshape((1, input_size_height, input_size_width)), keep_probability:1.0})
                for j in range(output_classes):
                    output[(0, j)] = max(output[(0, j)], output_temp[(0, j)])

            for j in range(output_classes):
                test_predictions_file.write('{}_classnumber_{}, {}\n'.format(file_name, j + 1, output[(0, j)]) )

with open('train_predictions.csv', 'w') as test_predictions_file:
    test_predictions_file.write('ID,Probability\n')

    output = numpy.zeros((1, output_classes))

    index = 0
    activation = numpy.zeros((batch_size, output_classes))
    for i in range(len(images)):

        train_image = images[i]

        c = -1
        for j in range(output_classes):
            if labels[i][(0, j)] == 1.0:
                c = j+1

        output = numpy.zeros((1, output_classes))
        output = y_output_softmax.eval(feed_dict = {x_place: train_image.reshape((1, input_size_height, input_size_width)), keep_probability:1.0})

        for j in range(output_classes):
            test_predictions_file.write('train_image_class_{}_classnumber_{}_image_{}, {}\n'.format(c, j + 1, i, output[(0, j)]))
            activation[(index, j)] = output[(0, j)]

        index = (index + 1) % batch_size

        if index == 0:

            plot.imshow(activation, interpolation='nearest', cmap=plot.cm.Blues)
            plot.title("Activation matrix", fontsize=16)
            colorbar = plot.colorbar(fraction=0.046, pad=0.04)
            activation = numpy.zeros((batch_size, output_classes))
            matplotlib.pyplot.show()

print('Prediction is completed.')
