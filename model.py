import csv
import numpy as np
import matplotlib.image as mpimg

lines = []

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

print(X_train.shape)
print(y_train.shape)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Activation, MaxPooling2D, Dropout, GlobalAveragePooling2D
#from tensorflow.keras.applications import NASNetLarge
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers.schedules import ExponentialDecay

"""
# INCEPTION_V3
X_train = preprocess_input(X_train)
inception = InceptionV3(weights='imagenet', include_top=False, input_shape=(299,299,3))
inception.trainable = False
#print(inception.summary())

driving_input = Input(shape=(160,320,3))
resized_input = Lambda(lambda image: tf.image.resize(image,(299,299)))(driving_input)
inp = inception(resized_input)

x = GlobalAveragePooling2D()(inp)
x = Dense(512, activation = 'relu')(x)
result = Dense(1, activation = 'relu')(x)


model = Model(inputs = driving_input, outputs = result)
model.compile(optimizer='Adam', loss='mse', metrics = ['accuracy'])

checkpoint = ModelCheckpoint(filepath="./ckpts/model.ckpt", monitor='val_loss', save_best_only=True)
stopper = EarlyStopping(monitor='val_acc', min_delta=0.0003, patience = 10)

batch_size = 64
epochs = 100

model.fit(x=X_train, y=y_train, shuffle=True, validation_split=0.2, epochs=epochs, 
	batch_size=batch_size, verbose=1, callbacks=[checkpoint, stopper])

"""

"""
# NASNET
nasnet = NASNetLarge(weights='imagenet', include_top=False, input_shape = (331,331,3))
nasnet.trainable = False
nasnet_input = Input(shape=(160,320,3))
resized_input = Lambda(lambda image: tf.image.resize(image, (331,331)))(nasnet_input)
inp = nasnet(resized_input)
x = GlobalAveragePooling2D()(inp)
x = Dense(512, activation='relu')(x)
predictions = Dense(1, activation='relu')(x)

model = Model(inputs=nasnet_input, outputs=predictions)
model.compile(optimizer='Adam', loss='mse', metrics = ['accuracy'])

checkpoint = ModelCheckpoint(filepath="./ckpts/model.ckpt", monitor='val_loss', save_best_only=True)
stopper = EarlyStopping(monitor='val_acc', min_delta=0.0003, patience = 10)

batch_size = 32
epochs = 100

model.fit(X_train, y_train, shuffle=True, validation_split=0.2, epochs=epochs, 
	batch_size=batch_size, verbose=1, callbacks=[checkpoint, stopper])
"""

# CUSTOM


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

model = Sequential()
model.add(Conv2D(input_shape=(160, 320, 3), filters=32, kernel_size=3, padding="valid"))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Activation('relu'))

model.add(Conv2D(filters=256, kernel_size=3, padding="valid"))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Activation('relu'))

model.add(Conv2D(filters=512, kernel_size=3, padding="valid"))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(1))

checkpoint = ModelCheckpoint(filepath="./ckpts/model.ckpt", monitor='val_loss', save_best_only=True)
stopper = EarlyStopping(monitor='val_acc', min_delta=0.0003, patience = 10)

lr_schedule = ExponentialDecay(initial_learning_rate=1.0,
	decay_steps=10000, decay_rate=0.9)
optimizer = Adam(learning_rate=lr_schedule)
loss = Huber(delta=0.5, reduction="auto", name="huber_loss")
model.compile(loss = loss, optimizer = optimizer, metrics=['accuracy'])
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, 
			epochs = 100, callbacks=[checkpoint, stopper])



model.save('model.h5')
