
import numpy as np
import pandas as pd

##All the preprocessing code
def encode_label(Y): 
	'''
	'll encode labels {0-0,90-1,180-2,270-3}
	'''
	# turn Y into an indicator matrix for training
	for i in range(len(Y)):
	    if (Y[i]==90) : Y[i]=1
	    elif (Y[i]==180) : Y[i]=2
	    elif (Y[i]==270) : Y[i]=3


def getData(file): 

	'''
	Read the data-file and return 
	X:feature vector 
	Y:target vector 
	ids: id of image file
	''' 

	print("Vectorizing " + '' + file)
	Y = [] #orientations
	X = [] #Features 
	ids=[] #ids
	for line in open(file):   
	        row = line.split(' ') 
	        ids.append(row[0])
	        Y.append(int(row[1]))
	        X.append([int(p) for p in row[2:]])

	ids,X,Y = np.array(ids),np.array(X),np.array(Y) 
	encode_label(Y)
	return ids,X,Y

class neural_network(object):  
	'''
	implementation of neural_network  
	Three important parameters:

	hidden_count = no of neuron in hidden layer
	alpha = learning_rate
	iterations = no of iterations for gradient descent
	'''

	def __init__(self, hidden_count=3,alpha=10e-7,iterations=200):
		self.M = hidden_count  
		self.alpha = alpha
		self.iterations = iterations
		self.W1 = None
		self.W2 = None
		self.b1 = None
		self.b2 = None

	def derivative_w2(self,Z,T,Y):
		N, K = T.shape
		M = Z.shape[1] 
		ret4 = Z.T.dot(T - Y)
		return ret4

	def derivative_w1(self,X, Z, T, Y, W2):
	    N, D = X.shape
	    M, K = W2.shape

	    dZ = (T - Y).dot(W2.T) * Z * (1 - Z)
	    ret2 = X.T.dot(dZ)

	    return ret2


	def derivative_b2(self,T, Y):
	    return (T - Y).sum(axis=0)


	def derivative_b1(self,T, Y, W2, Z):
	    return ((T - Y).dot(W2.T) * Z * (1 - Z)).sum(axis=0)


	def cost(self,T, Y):
	    tot = T * np.log(Y)
	    return tot.sum()  


	def predict(self,X):  
		'''
		Predicting probability of each class for each observation
		as per current parameter
		'''
		W1=self.W1 
		b1=self.b1
		W2=self.W2 
		b2=self.b2
		Z = 1 / (1 + np.exp(-X.dot(W1) - b1))
		A = Z.dot(W2) + b2
		expA = np.exp(A)
		Y = expA / expA.sum(axis=1, keepdims=True)
		return Y, Z
	  

	# determine the classification rate
	def classification_rate(self,Y, P):
	    n_correct = 0
	    n_total = 0
	    for i in xrange(len(Y)):
	        n_total += 1
	        if (Y[i] == P[i]):
	            n_correct += 1
	    return float(n_correct) / n_total

	def conf_matrix(self,label_dict,y_actual,y_predicted):   
		'''
		function to output confusion_matrix 
		'''
		actual =[]
		predicted = [] 

		for i in range(len(y_actual)):
			actual.append(label_dict[y_actual[i]])

		for i in range(len(y_predicted)):
			predicted.append(label_dict[y_predicted[i]])  

		actual = pd.Series(actual,name='Actual') 
		predicted = pd.Series(predicted,name='predicted')

		df_confusion = pd.crosstab(actual,predicted)   

		return df_confusion


	def make_file(self,label_dict,ids,predictions): 
		'''
		function to output nnet_output.txt
		'''
		f = open("nnet_output.txt","w")
		for i in range(len(predictions)):
			predictions[i]=label_dict[predictions[i]]
		for id,p in zip(ids,predictions):
			line = str(id) + " " + str(p)
			f.write(line)
			f.write("\n") 
		f.write("\n")
		f.close()


	def fit(self,X,Y):      
		'''
		function to train the neural_network
		'''
		D = len(X[0]) # dimensionality of input
		M = self.M # hidden layer size
		K = len(np.unique(Y)) # number of classes  
		N=len(Y) 
		T = np.zeros((N, K)) #one hot encoding
		for i in xrange(N):
		    T[i, Y[i]] = 1
	        

	    # Randomly initialize weights
		self.W1 = np.random.randn(D, M)
		self.b1 = np.random.randn(M)
		self.W2 = np.random.randn(M, K)
		self.b2 = np.random.randn(K)   

		learning_rate = self.alpha
		costs = [] 
		print ("Running:",self.iterations, "iterations of Gradient Ascent")
		i=0
		for epoch in xrange(self.iterations):
			output, hidden = self.predict(X)
			if (epoch % 100 == 0): 
				i=i+100
				c = self.cost(T, output)
				P = np.argmax(output, axis=1)
				r = self.classification_rate(Y, P)
				print ("iteration:",i,"cost:", c, "classification_rate:", r)
				# print "iteration:",i,"cost:", c			
				costs.append(c)

			# this is gradient ASCENT
			self.W2 += learning_rate * self.derivative_w2(hidden, T, output)
			self.b2 += learning_rate * self.derivative_b2(T, output)
			self.W1 += learning_rate * self.derivative_w1(X, hidden, T, output, self.W2)
			self.b1 += learning_rate * self.derivative_b1(T, output, self.W2, hidden)


