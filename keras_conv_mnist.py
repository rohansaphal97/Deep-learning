import matplotlib.pyplot as plt 
import tensorflow
import numpy as np
import math

from keras.models import Sequential,Model,load_model
from keras.layers import InputLayer,Input 
from keras.layers import Reshape,MaxPooling2D
from keras.layers import Conv2D,Dense,Flatten
from keras.optimizers import Adam



from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)

data.test.cls = np.argmax(data.test.labels, axis=1)

img_size=28
img_shape=(img_size,img_size)
img_shape_full=(img_size,img_size,1)
img_size_flat=img_size*img_size

num_channels=1

num_classes=10

def seq():

	model=Sequential()

	model.add(InputLayer(input_shape=(img_size_flat,)))

	model.add(Reshape(img_shape_full))

	model.add(Conv2D(kernel_size=5,filters=16,strides=1,padding='same',activation='relu',name='layer_conv1'))

	model.add(MaxPooling2D(pool_size=2,strides=2))

	model.add(Conv2D(kernel_size=5,filters=36,strides=1,padding='same',activation='relu',name='layer_conv2'))

	model.add(MaxPooling2D(pool_size=2,strides=2))

	model.add(Flatten())

	model.add(Dense(128,activation='relu'))

	model.add(Dense(num_classes,activation='softmax'))

	

	return model

model=seq()

optimizer=Adam(lr=1e-3)

model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(x=data.train.images,y=data.train.labels,epochs=5,batch_size=256)

result=model.evaluate(x=data.test.images,y=data.test.labels)

for name,value in zip(model.metrics_names,result):
	print(name,value)


def func():

	inputs=Input(shape=(img_size_flat,))

	net=Reshape(img_shape_full)(inputs)

	conv1=Conv2D(kernel_size=5,strides=1,filters=16,activation='relu',name='layer_conv1')(net)

	max1=MaxPooling2D(pool_size=2,strides=2)(conv1)

	conv2=Conv2D(kernel_size=5,strides=1,filters=32,activation='relu',name='layer_conv2')(max1)

	max2=MaxPooling2D(pool_size=2,strides=2)(conv2)

	flat=Flatten()(max2)

	dense1=Dense(128,activation='relu')(flat)

	output=Dense(10,activation='softmax')(dense1)

	model1=Model(inputs=inputs,output=output)

	return model1

model1=func()

model1.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

model1.fit(x=data.train.images,y=data.train.labels,epochs=5,batch_size=256)

result1=model1.evaluate(x=data.test.images,y=data.test.labels)

for name,value in zip(model1.metrics_names,result1):
	print(name,value)

path_model='keras_mnist_functional.keras'
model.save(path_model)

def load():

	model3=load_model(path_model)

	result1=model3.evaluate(x=data.test.images,y=data.test.labels)

	for name,value in zip(model3.metrics_names,result1):
		print(name,value)

		
