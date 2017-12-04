import sys
from dataLoader import dataLoader
from neuralnet import neuralnet

if __name__ == '__main__':
	dataLoader = dataLoader()
	neuralnet = neuralnet()
	#X_train, Y_train, X_test, Y_test = dataLoader.load_mnist()
	#parameters = neuralnet.model(X_train.T, Y_train.T, X_test.T, Y_test.T)


	X_train, Y_train = dataLoader.get_data_set("train")
	X_test, Y_test = dataLoader.get_data_set("test")
	parameters = neuralnet.model(X_train.T, Y_train.T, X_test.T, Y_test.T)
