from naive_convnet import get_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def predict(img_path):

	print "Loading model and weights ... "
	model = get_model()
	model.load_weights("model.h5")
	print "Loaded"

	print "compilling model"
	model.compile(loss='binary_crossentropy',
	              optimizer='rmsprop',
	              metrics=['accuracy'])

	img = load_img(img_path, target_size=(150, 150))
	# this is a Numpy array with shape (3, 150, 150)
	x = img_to_array(img)  
	x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
	print "loaded image of size: "+str(x.shape)
	res = model.predict_proba(x)
	print "result: "+str(res)


if __name__=="__main__":
	img_path = 'data/train/cats/cat.0.jpg'
	predict(img_path)

