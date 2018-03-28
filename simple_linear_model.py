import matplotlib.pyplot as plt
import tensorflow as tf 
import numpy as np 
from sklearn.metrics import confusion_matrix

from tensorflow.examples.tutorials.mnist import input_data

#Reading the MNIST dataset

data=input_data.read_data_sets("data/MNIST/",one_hot=True)

#Checking size of the train, validation and test.

print 'size of train, validation and test respectively is:'
print len(data.train.labels)
print len(data.validation.labels)
print len(data.test.labels)

#storing the labels in integer form, converting from one-hot encoded state

data.test.cls=np.array([label.argmax() for label in data.test.labels])

print data.test.cls[0:8]

#size of each image
image_size=28

#flatteneing the image and storing it in a 1D form
flattened_size=image_size*image_size

#defining the image shape
image_shape=(image_size,image_size)

#defining the number of classes in the classification process
classes=10

def plot_images(images, true_class, pred_class=None):

	assert len(images)==len(true_class)==9
	fig, axes=plt.subplots(3,3)
	fig.subplots_adjust(hspace=0.3, wspace=0.3)

	# print axes.flat

	for i, ax in enumerate(axes.flat):

		ax.imshow(images[i].reshape(image_shape),cmap='binary')

		if pred_class is None:
			xlabel = "True:{0}".format(true_class[i])
        else:
        	xlabel = "True: {0}, Pred: {1}".format(true_class[i], pred_class[i])

        ax.set_xlabel(xlabel)

        ax.set_xticks([])
        ax.set_yticks([])

	plt.show()

images=data.train.images[0:9]
true_class=data.train.labels[0:9]

# plot_images(images,true_class)


#tensorflow code for linear model starts here

x=tf.placeholder(tf.float32,[None,flattened_size])
y_true=tf.placeholder(tf.float32,[None,classes])
y_true_class=tf.placeholder(tf.int64,[None])

weights=tf.Variable(tf.zeros([flattened_size,classes]))
biases=tf.Variable(tf.zeros([classes]))

output = tf.matmul(x,weights) + biases
y_predicted=tf.nn.softmax(output)
y_predicted_class=tf.argmax(y_predicted,axis=1)

cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=y_true)
cost=tf.reduce_mean(cross_entropy)

optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)

correct_prediction=tf.equal(y_predicted_class,y_true_class)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session=tf.Session()

# session.run(cross_entropy)

session.run(tf.global_variables_initializer())
batch_size=10000


def optimization(n_iter):

	for i in range(n_iter):

		x_batch,y_true_batch=data.train.next_batch(batch_size)
		feed_dict_train={x:x_batch,y_true:y_true_batch}
		session.run(optimizer,feed_dict_train)


feed_dict_test={x:data.test.images,y_true:data.test.labels,y_true_class:data.test.cls}

def print_acc():

	acc=session.run(accuracy,feed_dict_test)

	print("Accuracy on test-set: {0:.1%}".format(acc))


def plot_weights():

	w=session.run(weights)

	w_min=np.min(w)
	w_max=np.max(w)

	fig, axes = plt.subplots(3, 4)
	fig.subplots_adjust(hspace=0.3, wspace=0.3)

	for i, ax in enumerate(axes.flat):
        # Only use the weights for the first 10 sub-plots.

		if i<10:
            # Get the weights for the i'th digit and reshape it.
            # Note that w.shape == (img_size_flat, 10)
			image = w[:, i].reshape(img_shape)

            # Set the label for the sub-plot.
			ax.set_xlabel("Weights: {0}".format(i))

			# Plot the image.
			ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')

			# Remove ticks from each sub-plot.
			ax.set_xticks([])
			ax.set_yticks([])
	plt.show()


optimization(10)
print_acc()
plot_weights()



