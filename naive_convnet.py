from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import tensorflow as tf
import os
from keras.utils.visualize_util import plot

tf.python.control_flow_ops = tf

# input: (150, 150, 3) matrix - 150*150 image with RGB 
# output: 1/0
def get_model():
	model = Sequential()
	model.add(Convolution2D(32, 3, 3, input_shape=(150, 150, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(32, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(64, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	# the model so far outputs 3D feature maps (height, width, features)

	model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
	model.add(Dense(64))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.compile(loss='binary_crossentropy',
	              optimizer='rmsprop',
	              metrics=['accuracy'])

	return model

def train():
	model = get_model()

	# this is the augmentation configuration we will use for training
	train_datagen = ImageDataGenerator(
	        rescale=1./255,
	        shear_range=0.2,
	        zoom_range=0.2,
	        horizontal_flip=True)

	# this is the augmentation configuration we will use for testing:
	# only rescaling
	test_datagen = ImageDataGenerator(rescale=1./255)

	# this is a generator that will read pictures found in
	# subfolers of 'data/train', and indefinitely generate
	# batches of augmented image data
	train_generator = train_datagen.flow_from_directory(
	        'data/train',  # this is the target directory
	        target_size=(150, 150),  # all images will be resized to 150x150
	        batch_size=32,
	        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

	# this is a similar generator, for validation data
	validation_generator = test_datagen.flow_from_directory(
	        'data/validation',
	        target_size=(150, 150),
	        batch_size=32,
	        class_mode='binary')


	model.fit_generator(
	        train_generator,
	        samples_per_epoch=1999,
	        nb_epoch=20,
	        validation_data=validation_generator,
	        nb_val_samples=800)

	## post processing
	# 1. save weights
	model.save_weights('model.h5')  # always save your weights after training or during training
	# 2. plot model
	plot(model, to_file='model.png')
	# 3. save model as json
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
		json_file.write(model_json)



if __name__=="__main__":
	img_width, img_height = 150, 150
	train_data_dir = 'data/train'
	validation_data_dir = 'data/validation'
	nb_train_samples = 2000
	nb_validation_samples = 800
	nb_epoch = 50


	train()