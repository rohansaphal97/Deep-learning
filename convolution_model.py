import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math

f1=5
nf1=16

f2=5
nf2=36

fc=128


from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)
data.test.cls = np.argmax(data.test.labels, axis=1)

img_size=28

flattened_image=img_size*img_size

img_shape=(img_size,img_size)

channels=1
classes=10


def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def conv_layer(input,channels,filter_size,num_filters,pooling=True):

	shape=[filter_size,filter_size,channels,num_filters]
	print shape

	w=new_weights(shape)
	b=new_biases(num_filters)

	layer=tf.nn.conv2d(input,filter=w,strides=[1,1,1,1],padding='SAME')
	layer+=b

	if pooling:
		layer=tf.nn.max_pool(value=layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
	else:
		layer=layer

	layer=tf.nn.relu(layer)

	return layer,w

def flatten_layer(layer):

	layer_shape=layer.get_shape()

	print layer_shape

	num_features=layer_shape[1:4].num_elements()

	layer_flat=tf.reshape(layer,[-1,num_features])

	return layer_flat,num_features

def fc_layer(input,num_inputs,num_outputs,use_relu=True):

	w=new_weights(shape=[num_inputs,num_outputs])
	b=new_biases(length=num_outputs)

	layer=tf.matmul(input,w)+b

	if use_relu:
		layer=tf.nn.relu(layer)


	return layer


x = tf.placeholder(tf.float32, shape=[None, flattened_image], name='x')

x_image = tf.reshape(x, [-1, img_size, img_size, channels])

y_true = tf.placeholder(tf.float32, shape=[None, classes], name='y_true')

y_true_cls = tf.argmax(y_true, axis=1)

layer_conv1, weights_conv1 = conv_layer(input=x_image,channels=channels,filter_size=f1,num_filters=nf1,pooling=True)
layer_conv2, weights_conv2 = conv_layer(input=layer_conv1,channels=nf1,filter_size=f2,num_filters=nf2,pooling=True)
layer_flat, num_features = flatten_layer(layer_conv2)

layer_fc1 =fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc,
                         use_relu=True)

layer_fc2 = fc_layer(input=layer_fc1,
                         num_inputs=fc,
                         num_outputs=classes,
                         use_relu=False)


y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, axis=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)

cost = tf.reduce_mean(cross_entropy)


optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()

session.run(tf.global_variables_initializer())


train_batch_size = 64

# total_iterations = 0
def optimize(num_iterations):

	global total_iterations

	start_time = time.time()

	for i in range(num_iterations):

		x_batch, y_true_batch = data.train.next_batch(train_batch_size)

		feed_dict_train = {x: x_batch, y_true: y_true_batch}

		session.run(optimizer, feed_dict=feed_dict_train)

		if i % 100 == 0:

			acc = session.run(accuracy, feed_dict=feed_dict_train)

			print("Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}".format(i + 1, acc))

	end_time = time.time()
	time_dif = end_time - start_time
	print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


test_batch_size = 256

def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

	num_test = len(data.test.images)

	cls_pred = np.zeros(shape=num_test, dtype=np.int)

	i = 0

	while i < num_test:
	# The ending index for the next batch is denoted j.
		j = min(i + test_batch_size, num_test)

		# Get the images from the test-set between index i and j.
		images = data.test.images[i:j, :]

		# Get the associated labels.
		labels = data.test.labels[i:j, :]

		# Create a feed-dict with these images and labels.
		feed_dict = {x: images,y_true: labels}

		# Calculate the predicted class using TensorFlow.
		cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

		# Set the start-index for the next batch to the
		# end-index of the current batch.
		i = j

	cls_true = data.test.cls

	correct = (cls_true == cls_pred)

	correct_sum = correct.sum()

	acc = float(correct_sum) / num_test

	msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
	print(msg.format(acc, correct_sum, num_test))



print_test_accuracy()

optimize(num_iterations=10000)


print_test_accuracy()





















