from naive_convnet import get_model

def predict():
	print "Loading model and weights ... "
	model = get_model()
	model.load_weights("model.h5")
	print "Loaded"



if __name__ == "main":
	predict()

