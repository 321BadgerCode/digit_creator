import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pickle

class ActivationFunctions:
	@staticmethod
	def sigmoid(x):
		return 1 / (1 + np.exp(-x))

	@staticmethod
	def sigmoidDerivative(x):
		return x * (1 - x)

	@staticmethod
	def relu(x):
		return np.maximum(0, x)

	@staticmethod
	def reluDerivative(x):
		return np.where(x > 0, 1, 0)

class LossFunctions:
	@staticmethod
	def meanSquaredError(yTrue, yPred):
		return np.mean((yTrue - yPred) ** 2)

	@staticmethod
	def meanSquaredErrorDerivative(yTrue, yPred):
		return yPred - yTrue

class NeuralLayer:
	def __init__(self, numInputs, numNeurons, activationFunc):
		self.weights = np.random.randn(numInputs, numNeurons) * 0.1
		self.biases = np.zeros((1, numNeurons))
		self.activationFunc = activationFunc
		self.input = None
		self.output = None
		self.z = None

	def forward(self, inputData):
		self.input = inputData
		self.z = np.dot(inputData, self.weights) + self.biases
		self.output = self.activationFunc(self.z)

	def backward(self, outputError, learningRate):
		activationDerivative = ActivationFunctions.sigmoidDerivative(self.output) if self.activationFunc == ActivationFunctions.sigmoid else ActivationFunctions.reluDerivative(self.output)
		delta = outputError * activationDerivative
		inputError = np.dot(delta, self.weights.T)
		weightsGradient = np.dot(self.input.T, delta)

		self.weights -= learningRate * weightsGradient
		self.biases -= learningRate * np.mean(delta, axis=0, keepdims=True)
		return inputError

class NeuralNetwork:
	def __init__(self):
		self.layers = []
		self.lossFunction = LossFunctions.meanSquaredError
		self.lossDerivative = LossFunctions.meanSquaredErrorDerivative

	def addLayer(self, layer):
		self.layers.append(layer)

	def forward(self, inputData):
		output = inputData
		for layer in self.layers:
			layer.forward(output)
			output = layer.output
		return output

	def backward(self, yTrue, yPred, learningRate):
		error = self.lossDerivative(yTrue, yPred)
		for layer in reversed(self.layers):
			error = layer.backward(error, learningRate)

	def train(self, xTrain, yTrain, epochs, learningRate):
		for epoch in range(epochs):
			output = self.forward(xTrain)
			loss = self.lossFunction(yTrain, output)
			self.backward(yTrain, output, learningRate)
			if epoch % 100 == 0:
				print(f"Epoch {epoch}/{epochs}, Loss: {loss}")

	def predict(self, inputData):
		return self.forward(inputData)

	def save(self, filename):
		with open(filename, "wb") as file:
			pickle.dump(self, file)

	def load(self, filename):
		with open(filename, "rb") as file:
			model = pickle.load(file)
			self.layers = model.layers

class DigitToImageNetwork:
	def __init__(self, inputSize, outputSize):
		self.network = NeuralNetwork()

		self.network.addLayer(NeuralLayer(inputSize, 128, ActivationFunctions.relu))
		self.network.addLayer(NeuralLayer(128, 256, ActivationFunctions.relu))
		self.network.addLayer(NeuralLayer(256, outputSize, ActivationFunctions.sigmoid))  # Output layer for image

	def forward(self, inputData):
		return self.network.forward(inputData)

	def train(self, xTrain, yTrain, epochs, learningRate):
		for epoch in range(epochs):
			output = self.network.forward(xTrain)
			loss = LossFunctions.meanSquaredError(yTrain, output)
			error = LossFunctions.meanSquaredErrorDerivative(yTrain, output)
			self.network.backward(yTrain, output, learningRate)
			if epoch % 100 == 0:
				print(f"Epoch {epoch}/{epochs}, Loss: {loss}")

	def generateImage(self, inputData):
		return self.network.forward(inputData)

def one_hot_encode(labels, numClasses):
	return np.eye(numClasses)[labels]

if __name__ == "__main__":
	# Set seed for reproducibility
	np.random.seed(0)
	print(f"Seed: {np.random.get_state()[1][0]}")

	# Check if model exists
	if not os.path.exists("./model.pkl"):
		# Example: Using MNIST-like data
		# from sklearn.datasets import load_digits
		# digits = load_digits()
		# yTrain = digits.data / 16.0  # Normalize pixel values
		# labels = digits.target

		# Example: Using image data in ./digits/<font>/<0-9>.png
		yTrain = []
		labels = []
		for font in os.listdir("./digit/"):
			for digit in os.listdir(f"./digit/{font}/"):
				image = Image.open(f"./digit/{font}/{digit}")
				image = np.array(image)
				image = image[:, :, 3]
				yTrain.append(image.flatten() / 255.0)
				labels.append(int(digit.split(".")[0]))
		yTrain = np.array(yTrain)
		labels = np.array(labels)

		inputSize = 10  # One-hot encoded vector size for digits 0-9
		outputSize = yTrain.shape[1]  # 64 for 8x8 images

		# Create one-hot encoded input data
		xTrain = one_hot_encode(labels, inputSize)

		# Initialize the network
		digitToImageNet = DigitToImageNetwork(inputSize, outputSize)

		# Train the network
		digitToImageNet.train(xTrain, yTrain, epochs=1000, learningRate=0.01)

		# Save the model
		digitToImageNet.network.save("./model.pkl")
	else:
		# Load the model
		digitToImageNet = DigitToImageNetwork(10, 64)
		digitToImageNet.network.load("./model.pkl")
	
	# Test the network with example digits
	generated_images = [np.zeros((8, 8)) for _ in range(10)]
	for i in range(len(generated_images)):
		test_digit = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(1, -1)  # One-hot encoding for "3"
		test_digit[0][i] = 1
		generated_image = digitToImageNet.generateImage(test_digit)
		generated_images[i] = generated_image

	# Display the generated images
	# fig, axes = plt.subplots(2, 10, figsize=(10, 2))
	# fig.suptitle("Training Data (Top) vs. Generated Images (Bottom)", fontsize=16)
	# fig.subplots_adjust(top=.75)
	# for i in range(10):
	# 	axes[0, i].imshow(yTrain[i].reshape(8, 8), cmap='gray')
	# 	axes[0, i].axis('off')
	# 	axes[0, i].set_title(str(labels[i]))
	# 	axes[1, i].imshow(generated_images[i].reshape(8, 8), cmap='gray')
	# 	axes[1, i].axis('off')
	fig, axes = plt.subplots(1, 10, figsize=(10, 2))
	fig.suptitle("Generated Images", fontsize=16)
	fig.subplots_adjust(top=.75)
	for i in range(10):
		axes[i].imshow(generated_images[i].reshape(8, 8), cmap="gray")
		axes[i].axis("off")
		axes[i].set_title(str(i))
	plt.show()