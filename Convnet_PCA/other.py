import keras
import matplotlib.pyplot as plt
from keras import optimizers, Model
from keras.datasets import mnist
from keras.layers import Dense, Activation, K, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import plot_model
import numpy as np


def learn_linear(x_train,y_train,x_test,y_test):

	num_pixels = x_train.shape[1] * x_train.shape[2]
	# normalize images
	x_train = x_train.reshape(x_train.shape[0], num_pixels).astype('float32')
	x_test = x_test.reshape(x_test.shape[0], num_pixels).astype('float32')
	x_train /= 255
	x_test /= 255
	# convert class vectors to binary class matrices

	y_train = keras.utils.to_categorical(y_train, 10)
	y_test = keras.utils.to_categorical(y_test, 10)

	model = Sequential()
	model.add(Dense(10,input_dim=num_pixels, kernel_initializer='normal', activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['acc'])
	plot_model(model,to_file='linear_model.png')
	m = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=64, verbose=2)

	plt.plot(m.history['acc'])
	plt.plot(m.history['val_acc'])
	plt.title('learn linear MNIST accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.savefig("learn_linear_MNIST_accuracy.png")
	plt.show()

	plt.plot(m.history['loss'])
	plt.plot(m.history['val_loss'])
	plt.title('learn linear MNIST loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.savefig("learn_linear_MNIST_loss.png")
	plt.show()


def learn_MCP(x_train,y_train,x_test,y_test):
	num_pixels = x_train.shape[1] * x_train.shape[2]
	x_train = x_train.reshape(x_train.shape[0], num_pixels).astype('float32')
	x_test = x_test.reshape(x_test.shape[0], num_pixels).astype('float32')
	x_train /= 255
	x_test /= 255
	y_train = keras.utils.to_categorical(y_train, 10)
	y_test = keras.utils.to_categorical(y_test, 10)

	model = Sequential()
	model.add(Dense(num_pixels, input_dim=num_pixels))
	model.add(Activation('relu'))
	model.add(Dense(num_pixels, input_dim=num_pixels))
	model.add(Activation('relu'))
	# final layer
	model.add(Dense(10, input_dim=num_pixels, kernel_initializer='normal', activation='softmax'))

	model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['acc'])
	plot_model(model,to_file='MCP_model.png')
	m = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=64, verbose=2)
	plt.plot(m.history['acc'])
	plt.plot(m.history['val_acc'])
	plt.title('learn MCP MNIST accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.savefig("learn_MCP_MNIST_accuracy.png")
	plt.show()

	plt.plot(m.history['loss'])
	plt.plot(m.history['val_loss'])
	plt.title('learn MCP MNIST loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.savefig("learn_MCP_MNIST_loss.png")
	plt.show()


def learn_cnn(x_train,y_train,x_test,y_test):
	num_classes = 10

	# input image dimensions
	img_rows, img_cols = 28, 28

	if K.image_data_format() == 'channels_first':
		x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
		x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
		input_shape = (1, img_rows, img_cols)
	else:
		x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
		x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
		input_shape = (img_rows, img_cols, 1)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255

	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(10, activation='softmax'))
	model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.sgd(),metrics=['acc'])
	plot_model(model,to_file='cnn_model.png')
	m = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=64, verbose=2)
	plt.plot(m.history['acc'])
	plt.plot(m.history['val_acc'])
	plt.title('learn cnn MNIST accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.savefig("learn_cnn_MNIST_accuracy.png")
	plt.show()

	plt.plot(m.history['loss'])
	plt.plot(m.history['val_loss'])
	plt.title('learn cnn MNIST loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.savefig("learn_cnn_MNIST_loss.png")
	plt.show()


def learning_rate_explore():
		(x_train, y_train), (x_test, y_test) = mnist.load_data()

		learn_rates = [0.0001,0.001,0.01,0.1,1,10]

		accuracy_history = []
		lose_history = []
		num_pixels = x_train.shape[1] * x_train.shape[2]
		x_train = x_train.reshape(x_train.shape[0], num_pixels).astype('float32')
		x_test = x_test.reshape(x_test.shape[0], num_pixels).astype('float32')
		x_train /= 255
		x_test /= 255

		y_train = keras.utils.to_categorical(y_train, 10)
		y_test = keras.utils.to_categorical(y_test, 10)

		for i in range(len(learn_rates)):

			model = Sequential()
			model.add(Dense(num_pixels, input_dim=num_pixels))
			model.add(Activation('relu'))
			model.add(Dense(num_pixels, input_dim=num_pixels))
			model.add(Activation('relu'))
			# final layer
			model.add(Dense(10, input_dim=num_pixels, kernel_initializer='normal', activation='softmax'))
			sgd = optimizers.SGD(lr=learn_rates[i])
			model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['acc'])
			m = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100	, batch_size=64,
						  verbose=2)

			accuracy_history.append([m.history['acc'],m.history['val_acc']])
			lose_history.append([m.history['loss'],m.history['val_loss']])

		colors = ['r','g','b','m','c','k']
		for j in range(len(accuracy_history)):
			plt.plot(accuracy_history[j][0],color=colors[j],linestyle='-',label='train'+str(learn_rates[j]))
			plt.plot(accuracy_history[j][1],color=colors[j],linestyle='--',label='valdation '+str(learn_rates[
																								   j]))
		plt.legend(loc='best')
		plt.ylabel('Accuracy')
		plt.xlabel('Epoch')
		plt.show()
		for j in range(len(lose_history)):
			plt.plot(lose_history[j][0],color=colors[j],linestyle='-',label='train'+str(learn_rates[j]))
			plt.plot(lose_history[j][1],color=colors[j],linestyle='--',label='valdation '+str(learn_rates[
																								   j]))
		plt.legend(loc='best')
		plt.ylabel('Lose')
		plt.xlabel('Epoch')
		plt.show()


def autoencoder(x_train,y_train):
	x_train = x_train.reshape(60000, 784) / 255

	model = Sequential()
	model.add(Dense(512, activation='relu', input_shape=(784,)))
	model.add(Dense(256, activation='relu'))
	model.add(Dense(128, activation='relu'))
	model.add(Dense(2, activation='linear', name="inner_layer"))
	model.add(Dense(128, activation='relu'))
	model.add(Dense(256, activation='relu'))
	model.add(Dense(512, activation='relu'))
	model.add(Dense(784, activation='sigmoid'))

	model.compile(loss='mean_squared_error', optimizer=Adam(), metrics=['mse'])
	plot_model(model,to_file='autoencoder_model.png')

	m = model.fit(x_train, x_train,  epochs=1, batch_size=128, verbose=1)

	encoder = Model(model.input, model.get_layer('inner_layer').output)
	Zenc = encoder.predict(x_train)  # bottleneck representation
	plt.title('Autoencoder')
	plt.scatter(Zenc[:5000, 0], Zenc[:5000, 1], c=y_train[:5000], cmap=plt.get_cmap('jet', 10), s=5)
	plt.colorbar()
	plt.gca().get_xaxis().set_ticklabels([])
	plt.gca().get_yaxis().set_ticklabels([])
	plt.show()

def PCA_encoder(X_train,Y_train):
	""""
	this code was taken from here:
	https://medium.com/analytics-vidhya/journey-from-principle-component-analysis-to-autoencoders-e60d066f191a
	"""
	x_train = X_train.reshape(60000, 784) / 255
	mu = x_train.mean(axis=0)
	U, s, V = np.linalg.svd(x_train - mu, full_matrices=False)
	Zpca = np.dot(x_train - mu, V.transpose())
	plt.title('PCA')
	plt.scatter(Zpca[:5000, 0], Zpca[:5000, 1], c=Y_train[:5000], cmap=plt.get_cmap('jet', 10), s=5)
	plt.colorbar()
	plt.gca().get_xaxis().set_ticklabels([])
	plt.gca().get_yaxis().set_ticklabels([])
	plt.show()



if __name__ == '__main__':
	(X_train,Y_train), (X_test, Y_test) = mnist.load_data()

	# learn_linear(X_train,Y_train,X_test,Y_test)
	# 	learn_MCP(X_train,Y_train,X_test,Y_test)
	# learn_cnn(X_train,Y_train,X_test,Y_test)
	# learning_rate_explore()
	# autoencoder(X_train,Y_train)
	# PCA_encoder(X_train,Y_train)