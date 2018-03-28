import matplotlib.pyplot as plt 
import tensorflow as tf 
import numpy as np 
from sklearn.metrics import confusion_matrix 
import math

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)

data.test.cls=np.argmax(data.test.labels,axis=1)

img_size=28
img_flattened_size=img_size*img_size
img_shape=(img_size,img_size)
channels=1
classes=10

x=tf.placeholder(tf.float32,shape=[None,img_flattened_size],name='x')
x_image=tf.reshape(x,[-1,img_size,img_size,channels])
y_true=tf.placeholder(tf.float32,shape=[None,classes],name='y_true')
y_true_cls=tf.argmax(y_true,dimension=1)

layer1=x_image
layer1=tf.layers.conv2d(inputs=layer1,name='layer_1',padding='same',filters=16,kernel_size=5,activation=tf.nn.relu)
layer1=tf.layers.max_pooling2d(inputs=layer1,pool_size=2,strides=2)
layer2=tf.layers.conv2d(inputs=layer1,name='layer_2',padding='same',filters=32,kernel_size=5,activation=tf.nn.relu)
layer2=tf.layers.max_pooling2d(inputs=layer2,pool_size=2,strides=2)
flat=tf.contrib.layers.flatten(layer2)
dense1=tf.layers.dense(inputs=flat,name='dense_1',units=128,activation=tf.nn.relu)
dense2=tf.layers.dense(inputs=dense1,name='dense_2',units=classes,activation=tf.nn.relu)
logits=dense2

y_pred=tf.nn.softmax(logits=logits)

y_pred_cls=tf.argmax(y_pred,dimension=1)

cross_entropy_loss=tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=logits)

cost=tf.reduce_mean(cross_entropy_loss)

optimizer=tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

correct=tf.equal(y_pred_cls,y_true_cls)
accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))

def get_weights(layer):
	with tf.variable_scope(layer,reuse=True):
		variable=tf.get_variable('kernel')

	return variable



session=tf.Session()
session.run(tf.global_variables_initializer())

train_batch=64

def optimize(iteration):

	for i in range(iteration):

		x_batch,y_batch=data.train.next_batch(train_batch)
		feed_dict_train={x:x_batch,y_true:y_batch}
		session.run(optimizer,feed_dict=feed_dict_train)

		if i%100==0:

			acc=session.run(accuracy,feed_dict=feed_dict_train)
			msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
			print(msg.format(i + 1, acc))

test_batch_size=256
		
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

optimize(iteration=110)

print_test_accuracy()