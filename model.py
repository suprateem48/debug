import csv
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from time import time

lines = []

t1=time()

with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []

for line in lines:
	image = mpimg.imread("C:\\Users\\Dolphin48\\Jupyter Notebooks\\CarND-Behavioral-Cloning-P3-master\\data\\"+str(line[0]))
	images.append(image)
	measurements.append(float(line[3]))

X_train = np.array(images)
y_train = np.array(measurements)

# Preprocess: Augment using flipping, Crop Images

X_train_flipped = []
for image in X_train:
	X_train_flipped.append(np.fliplr(image))

X_train_flipped = np.array(X_train_flipped)
y_train_flipped = np.multiply(y_train , -1.0)

X_train = np.append(X_train, X_train_flipped, axis = 0)
y_train = np.append(y_train, y_train_flipped)

X_train_bottom = []
for image in X_train:
	X_train_bottom.append(image[int(image.shape[0]/2.5):,:])


X_train = np.array(X_train_bottom)

#X_train, y_train = shuffle(X_train, y_train)

#X_train = (X_train - X_train.mean()) / X_train.std()

print(X_train.shape)
print(y_train.shape)


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Conv2D, Activation, MaxPooling2D, Dropout, GlobalAveragePooling2D
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.metrics import RootMeanSquaredError


def inception_v3():

	from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
	import tensorflow.compat.v1 as tf
	tf.disable_v2_behavior()

	# INCEPTION_V3
	global X_train, y_train
	X_train = preprocess_input(X_train)
	inception = InceptionV3(weights='imagenet', include_top=False, input_shape=(299,299,3))
	#inception.trainable = False
	print(inception.summary())

	driving_input = Input(shape=(96,320,3))
	resized_input = Lambda(lambda image: __import__("tensorflow").image.resize(image,(299,299)))(driving_input)
	inp = inception(resized_input)

	x = GlobalAveragePooling2D()(inp)

	x = Dense(512, activation = 'relu')(x)
	x = Dense(256, activation = 'relu')(x)
	x = Dropout(0.25)(x)
	x = Dense(128, activation = 'relu')(x)
	x = Dense(64, activation = 'relu')(x)
	x = Dropout(0.25)(x)
	result = Dense(1, activation = 'linear')(x)

	lr_schedule = ExponentialDecay(initial_learning_rate=0.0001, decay_steps=100000, decay_rate=0.95)
	optimizer = Adam(learning_rate=lr_schedule)
	loss = Huber(delta=0.5, reduction="auto", name="huber_loss")
	model = Model(inputs = driving_input, outputs = result)
	t2 = time()
	model.compile(optimizer=optimizer, loss=loss)

	checkpoint = ModelCheckpoint(filepath="./ckpts/model_inc.h5", monitor='val_loss', save_best_only=True)
	stopper = EarlyStopping(monitor='val_loss', min_delta=0.0003, patience = 10)

	model.fit(x=X_train, y=y_train, shuffle=True, validation_split=0.2, epochs=100, 
		batch_size=32, verbose=1, callbacks=[checkpoint, stopper])

	model.load_weights('./ckpts/model_inc.h5')

	print("Time taken to train: {:.2f}s".format(time()-t2))

	model.save('model.h5')

	print("Total time taken: {:.2f}s".format(time()-t1))

def nasnet():

	from tensorflow.keras.applications import NASNetLarge
	from tensorflow.keras.applications.nasnet import preprocess_input

	# NASNET
	global X_train, y_train
	X_train = preprocess_nasnet(X_train)
	nasnet = NASNetLarge(weights='imagenet', include_top=False, input_shape = (331,331,3))
	nasnet.trainable = False
	print(nasnet.summary())
	nasnet_input = Input(shape=(96,320,3))
	resized_input = Lambda(lambda image: tf.image.resize(image, (331,331)))(nasnet_input)
	inp = nasnet(resized_input)
	x = GlobalAveragePooling2D()(inp)
	x = Dense(512, activation = 'relu')(x)
	x = Dense(256, activation = 'relu')(x)
	x = Dropout(0.25)(x)
	x = Dense(128, activation = 'relu')(x)
	x = Dense(64, activation = 'relu')(x)
	x = Dropout(0.25)(x)
	predictions = Dense(1, activation='linear')(x)

	model = Model(inputs=nasnet_input, outputs=predictions)
	lr_schedule = ExponentialDecay(initial_learning_rate=0.0001, decay_steps=100000, decay_rate=0.95)
	optimizer = Adam(learning_rate=lr_schedule)
	loss = Huber(delta=0.5, reduction="auto", name="huber_loss")

	t2 = time()
	model.compile(optimizer=optimizer, loss=loss)

	checkpoint = ModelCheckpoint(filepath="./ckpts/model_nasnet.h5", monitor='val_loss', save_best_only=True)
	stopper = EarlyStopping(monitor='val_loss', min_delta=0.0003, patience = 20)

	model.fit(X_train, y_train, shuffle=True, validation_split=0.2, epochs=100, 
		batch_size=32, verbose=1, callbacks=[checkpoint, stopper])

	model.load_weights('./ckpts/model_nasnet.h5')

	print("Time taken to train: {:.2f}s".format(time()-t2))

	model.save('model.h5')

	print("Total time taken: {:.2f}s".format(time()-t1))


def custom():

	# CUSTOM
	global X_train, y_train

	model = Sequential()
	model.add(Conv2D(input_shape=(96, 320, 3), filters=32, kernel_size=3, padding="same"))
	model.add(MaxPooling2D(pool_size=(3,3)))
	model.add(Activation('relu'))

	model.add(Conv2D(filters=64, kernel_size=3, padding="same"))
	model.add(MaxPooling2D(pool_size=(3,3)))
	model.add(Activation('relu'))

	model.add(Conv2D(filters=128, kernel_size=3, padding="same"))
	model.add(MaxPooling2D(pool_size=(3,3)))
	model.add(Activation('relu'))
	#model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.25))
	model.add(Dense(256))
	model.add(Activation('relu'))
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dropout(0.25))
	model.add(Dense(1))
	model.add(Activation('linear'))

	checkpoint = ModelCheckpoint(filepath="./ckpts/model_custom.h5", monitor='val_loss', save_best_only=True)
	stopper = EarlyStopping(monitor='val_loss', min_delta=0.0003, patience = 10)

	lr_schedule = ExponentialDecay(initial_learning_rate=0.1, decay_steps=100000, decay_rate=0.95)
	optimizer = Adam(learning_rate=lr_schedule)
	loss = Huber(delta=0.5, reduction="auto", name="huber_loss")
	t2 = time()
	model.compile(loss = loss, optimizer = optimizer)
	model.fit(X_train, y_train, validation_split = 0.2, shuffle = True,
				epochs = 100, callbacks=[checkpoint, stopper])

	model.load_weights('./ckpts/model_custom.h5')

	print("Time taken to train: {:.2f}s".format(time()-t2))

	model.save('model.h5')

	print("Total time taken: {:.2f}s".format(time()-t1))


# Run models
inception_v3()