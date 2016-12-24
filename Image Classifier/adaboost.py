import six
import random
import numpy as np
import math

class Stump(object):
	"""
	Storing each Stump along with its properties
	"""

	def __init__(self, point1, point2, alpha):
		self.point1 = point1
		self.point2 = point2
		self.alpha = alpha

def fit(data, stump_count):
	"""
	Fit the given into a model and return that model
	"""
	print("Generating Model..")
	model = {}

	# Train for each orientation
	for orientation in (['0', '90', '180', '270']):

		# Initialise weights
		weights = init_weights(data)

		# Generate all possible stumps (stump_count)
		for stump in range(stump_count):
			error = 0

			# Generate Model (Basically combination of points) / Classifier
			x1 = random.randrange(96)
			x2 = random.randrange(97, 192)

			# Generate the correct and incorrect records based on classifier
			correct, incorrect = get_decision(data, orientation, x1, x2)

			# Calculate the sum of error
			for each_file in incorrect:
				error += weights[each_file]

			# Re calculate weights
			error_weight = 0.5 / float(len(incorrect))
			correct_weight = 0.5 / float(len(correct))

			# Calculate Alpha for this classifier
			alpha = 0.5 * math.log(((1.0 - error) / error),2)

			# Store this classifier / Stump
			stump = Stump(x1, x2, alpha)

			# Store that model for testing
			if (orientation not in model):
				model[orientation] = []
			model[orientation].append(stump)
			
			# Re calulate weights
			for each_file in correct:
				weights[each_file] = correct_weight

			for each_file in incorrect:
				weights[each_file] = error_weight

	return model

def classify(model, test_data):
	"""
	Classify the test data based on the model provided
	"""
	print("Testing Model..")
	record_size = 0
	correct = 0
	result = {}
	confusion = {}
	output = {}

	# Read Test File
	with open(test_data) as raw_data:
		records = raw_data.readlines()

	# Classify each record
	for record in records:

		# Record Information
		record_size += 1
		data = record.split()
		file_name = data[0]
		orientation = data[1]
		matrix = np.asarray(data[2:], dtype=int)

		# Classify each record with the help of built model
		for train_orientataion, stumps in six.iteritems(model):
			total = 0
			for stump in stumps:

				if (matrix[stump.point1] >= matrix[stump.point2]):
					total += stump.alpha
				else:
					total -= stump.alpha

			result[train_orientataion] = total

		# Check the decision, whether if it predicted correctly or not
		decision = 0
		maximum = float('-inf')
		for key, value in six.iteritems(result):
			if (value > maximum):
				maximum = value
				decision = key

		output[file_name] = str(decision)

		if (decision == orientation):
			correct += 1

		if (orientation not in confusion): confusion[orientation] = {}
		if (decision not in confusion[orientation]): confusion[orientation][decision] = 0

		confusion[orientation][decision] += 1

	return 100 * (correct / float(record_size)), confusion, output

def init_weights(data):
	"""
	Initialise weights for the current dataset
	"""
	weights = {}
	weight = 1 / float(len(data))
	for item in six.iterkeys(data):
		weights[item] = weight

	if(len(weights) < 1): print("No Weights")
	return weights

def get_decision(data, orientation, point1, point2):
	"""
	Based on the model, classify the records
	"""
	correct = []
	incorrect = []

	for each_file in six.iterkeys(data):
		if (int(data[each_file][orientation][point1]) >= int(data[each_file][orientation][point2])):
			correct.append(each_file)
		else:
			incorrect.append(each_file)

	return np.array(correct), np.array(incorrect)