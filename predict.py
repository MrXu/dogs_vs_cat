from naive_convnet import get_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


class naive_prediction(object):
	model = None

	def __init__(self):
		self.load_model()

	def load_model(self):
		print "Loading model and weights ... "
		model = get_model()
		model.load_weights("model.h5")
		print "Loaded"

		print "compilling model"
		model.compile(loss='binary_crossentropy',
		              optimizer='rmsprop',
		              metrics=['accuracy'])

		self.model = model

	def predict(self, img_path):

		img = load_img(img_path, target_size=(150, 150))
		# this is a Numpy array with shape (3, 150, 150)
		x = img_to_array(img)  
		x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
		print "loaded image of size: "+str(x.shape)
		res = self.model.predict_proba(x)
		print "result: "+str(res)

	def predict_list(self, img_arr):
		for x in img_arr:
			self.predict(x)

if __name__=="__main__":
	img_path = 'data/train/cats/cat.0.jpg'

	img_arr = []
	for i in range(20):
		img_arr.append('data/train/cats/cat.'+str(i)+'.jpg')

	naive_pred = naive_prediction()
	naive_pred.predict_list(img_arr)


