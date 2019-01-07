"""# Import"""

import numpy as np
import matplotlib.pyplot as plt
import os
import gzip
from sklearn.metrics import confusion_matrix
import time

"""# Neural Network Implementation"""
class NeuralNet:    
    class Layer:
        def __init__(self, total_neurons, activation):
            self.total_neurons = total_neurons
            self.weights = 0
            self.A = 0
            self.Z = 0
            self.error = 0
            self.activation = activation
            
        def set_weights(self, weights):
            self.weights = weights
        def set_activation_function(self, activation_function):
            self.activation_function = activation_function
        def set_activation_function_derivative(self, activation_function_derivative):
            self.activation_function_derivative = activation_function_derivative
            
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.layer = [self.Layer(input_dim, "noactivation")]
        self.output_dim = output_dim
        self.gamma = 0.9
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.history = {'train_cost':[], 'validation_cost':[],'train_acc':[], 'validation_acc':[], 'L':[], 'time':[]}
        print("init")
       
    def add_fully_connected_layer(self, total_neurons, activation):
        self.layer.append(self.Layer(total_neurons, activation))
        
    def display_layers(self):
        for i in range(1, len(self.layer)):
            l = self.layer[i]
            print(l.weights.shape)

    def build(self, output_activation="relu", momentum='nomomentum', gamma=0.9, beta1=0.9, beta2=0.999):
        self.layer.append(self.Layer(self.output_dim, output_activation))
        self.momentum = momentum
        last = self.input_dim
        for i in range(1, len(self.layer)):
            current_layer_neurons = self.layer[i].total_neurons
            self.layer[i].set_weights(np.random.randn(current_layer_neurons, last + 1)*np.sqrt(2.0/(last+1)))
            if momentum == "poly" or momentum == "nag":
                self.layer[i].v = np.zeros((current_layer_neurons, last + 1))
                self.gradient_update = self.poly_momentum_gradient_update
                self.gamma = gamma
            elif momentum == "rms":
                self.layer[i].gradient_square_sum = np.zeros((current_layer_neurons, last + 1))
                self.gradient_update = self.rms_prop_momentum_gradient_update
                self.gamma = gamma
            elif momentum == "adam":
                self.layer[i].m = np.zeros((current_layer_neurons, last + 1))
                self.layer[i].v = np.zeros((current_layer_neurons, last + 1))
                self.gradient_update = self.adam_gradient_update
                self.beta1 = beta1
                self.beta2 = beta2
            else:
                self.gradient_update = self.no_momentum_gradient_update
                
            act = self.layer[i].activation
            if act == "sigmoid":
                self.layer[i].set_activation_function(self.sigmoid)
                self.layer[i].set_activation_function_derivative(self.sigmoid_derivative)
            elif act == "tanh":
                self.layer[i].set_activation_function(self.tanh_activation)
                self.layer[i].set_activation_function_derivative(self.tanh_derivative)
            elif act == "relu":
                self.layer[i].set_activation_function(self.relu)
                self.layer[i].set_activation_function_derivative(self.relu_derivative)
            elif act == "softmax":
                self.layer[i].set_activation_function(self.softmax)
                self.layer[i].set_activation_function_derivative(self.softmax_derivative)
                
            last = current_layer_neurons
        print("building")
        
    def sigmoid(self, Z):
        return (1.0/(1.0+np.exp(-Z)))
    def sigmoid_derivative(self, A):
        return A * (1 - A)
    def tanh_activation(self, Z):
        return np.tanh(Z)
    def tanh_derivative(self, A):
        return 1 - (A * A)
    def relu(self, Z):
        return np.maximum(Z,0)
    def relu_derivative(self, A):
        A[0] = A[0] * 0
        A[A<=0] = 0
        A[A>0] = 1
        return A
    def leaky_relu(self, Z, e=0.01):
        return np.maximum(Z,e*Z)
    def leaky_relu_derivative(self, A, e=0.01):
        A[0] = A[0] * 0
        A[A<=0] = e
        A[A>0] = 1
        return A
    def softmax(self, Z):
        shift = Z - np.max(Z, axis=0)
        n = np.exp(shift)
        d = np.sum(n, axis = 0)
        return n/d
    def softmax_derivative(self, A):
        return A * (1 - A)
    def noactivation(self, Z):
        return Z
    
    def train(self, epochs, data, train_labels, Y, validation_data=[], validation_labels=[], learning_rate=0.01, decay_rate=0, callFunc=None, callEvery=1, model=1):       
       
        last = time.time()
        train_data = np.copy(data)
        total_layers = len(self.layer)
        for i in range(epochs):
            if i % callEvery == 0:
                if callFunc != None:
                    callFunc(validation_data, validation_labels, model)
                
            if i % callEvery == 0:
                if self.output_dim == 1:
                    val_acc = self.check_accuracy_bin(validation_data, validation_labels, i)
                    tr_acc = self.check_accuracy_bin(data, train_labels, i)
                else:
                    val_acc = self.check_accuracy(validation_data, validation_labels, i)
                    tr_acc = self.check_accuracy(data, train_labels, i)
                self.history['train_acc'].append(tr_acc)
                self.history['validation_acc'].append(val_acc)
                print('iteration', i, 'tr_acc', tr_acc, 'val_acc', val_acc)
                #print((learning_rate*(1.0/(1.0 + decay_rate * i))))
                print('--------------------------------------------')
           
            out = self.forward(data)
            self.backward(Y)
            alpha = learning_rate*(1.0/(1.0 + decay_rate * i))
            
            self.gradient_update(alpha, data.shape[0], i+1)
            if self.output_dim == 1:
                self.logistic_loss_function(out, train_labels, history_label='train_cost')
                out = self.forward(validation_data)
                self.logistic_loss_function(out, validation_labels, history_label='validation_cost')
            else:
                self.cross_entropy_loss_function(out, train_labels, history_label='train_cost')
                out = self.forward(validation_data)
                self.cross_entropy_loss_function(out, validation_labels, history_label='validation_cost')
            
            end = time.time()
            self.history['time'].append((end-last))
            last = end

    def forward(self, data):
        
        data = np.asarray(data).T          # d x m
        self.layer[0].A = np.insert(data, 0, 1,axis = 0)
        for itr in range(1, len(self.layer)):
            if self.momentum == "nag":
                self.layer[itr].Z = ((self.layer[itr].weights - self.gamma * self.layer[itr].v) @ self.layer[itr-1].A)
            else:
                self.layer[itr].Z = (self.layer[itr].weights @ self.layer[itr-1].A)
            self.layer[itr].A = self.layer[itr].activation_function(self.layer[itr].Z)
            self.layer[itr].A = np.insert(self.layer[itr].A, 0, 1,axis = 0)
        return self.layer[len(self.layer)-1].A[1:]

    def backward(self, Y):
        self.layer[len(self.layer)-1].error = self.layer[len(self.layer)-1].A - Y
        for i in reversed(range(1, len(self.layer)-1)):
            if self.momentum == "nag":          
                self.layer[i].error = ((self.layer[i+1].weights - self.gamma * self.layer[i+1].v).T @ self.layer[i+1].error[1:]) * self.layer[i].activation_function_derivative(self.layer[i].A)
            else:
                self.layer[i].error = (self.layer[i+1].weights.T @ self.layer[i+1].error[1:]) * self.layer[i].activation_function_derivative(self.layer[i].A)
        return
    
    def no_momentum_gradient_update(self, alpha, sample_size, t):
        for i in range(1, len(self.layer)):
            gradient = (self.layer[i].error[1:] @ self.layer[i-1].A.T)/sample_size
            self.layer[i].weights = self.layer[i].weights - (alpha * gradient)
       
    #NAG and poly has same momentum formula
    def poly_momentum_gradient_update(self, alpha, sample_size, t):
        for i in range(1, len(self.layer)):
            gradient = (self.layer[i].error[1:] @ self.layer[i-1].A.T)/sample_size
            self.layer[i].v = (self.gamma * self.layer[i].v) + (alpha * gradient)
            self.layer[i].weights = self.layer[i].weights - self.layer[i].v
            
    def rms_prop_momentum_gradient_update(self, alpha, sample_size, t):
        for i in range(1, len(self.layer)):
            gradient = (self.layer[i].error[1:] @ self.layer[i-1].A.T)/sample_size
            self.layer[i].gradient_square_sum = (self.gamma * self.layer[i].gradient_square_sum) + ((1-self.gamma) * (gradient * gradient))
            self.layer[i].weights = self.layer[i].weights - ((alpha/np.sqrt(self.layer[i].gradient_square_sum + 1e-8)) * gradient)
            
    def adam_gradient_update(self, alpha, sample_size, t):
        for i in range(1, len(self.layer)):
            gradient = (self.layer[i].error[1:] @ self.layer[i-1].A.T)/sample_size
            self.layer[i].m = (self.beta1 * self.layer[i].m) + ((1-self.beta1) * gradient)
            self.layer[i].v = (self.beta2 * self.layer[i].v) + ((1-self.beta2) * (gradient * gradient))
            m_corrected = self.layer[i].m / (1 - (self.beta1**t))
            v_corrected = self.layer[i].v / (1 - (self.beta2**t))
            self.layer[i].weights = self.layer[i].weights - (alpha/(np.sqrt(v_corrected) + 1e-8)) * m_corrected
    
    def predict(self, data):
        return self.forward([data])
        
    def check_accuracy(self, data, labels, j):
        incorrect = 0
        for i in range(data.shape[0]):
            result = self.predict(data[i])
            predicted_label = np.argmax(result)
            if labels[i] != predicted_label:
                incorrect = incorrect + 1
        return (1-((incorrect/1.0)/data.shape[0]))
    
    def check_accuracy_bin(self, data, labels, j):
        incorrect = 0        
        for i in range(data.shape[0]):
            result = self.predict(data[i])
            predicted_label = int(result[0][0]/0.5)
            if labels[i] != predicted_label:
                incorrect = incorrect + 1
        #print(j, 1-((incorrect/1.0)/data.shape[0]))
        return (1-((incorrect/1.0)/data.shape[0]))
    
    def logistic_loss_function(self, output, Y, history_label='train_cost'):
        sum = 0
        #print(Y.shape, output.shape)
        for sample in range(output.shape[1]):
            sum += int(Y[sample]) * np.log(output[0, sample]) + (1-int(Y[sample])) * (np.log(1-output[0, sample]))
        sum = -sum / output.shape[1]
        self.history[history_label].append(sum)
    def get_loss(self, output, Y, history_label='train_cost'):
        sum = 0
        for sample in range(output.shape[1]):
            sum += np.log(output[int(Y[sample]), sample] + 1e-10)
        sum = -sum / output.shape[1]
        return sum
    
    def cross_entropy_loss_function(self, output, Y, history_label='train_cost'):
        sum = 0
        for sample in range(output.shape[1]):
            sum += np.log(output[int(Y[sample]), sample] + 1e-10)
        sum = -sum / output.shape[1]
        self.history[history_label].append(sum)
    
    def get_layer(self, num):
        return self.layer[num]

		
"""# Data Load Methods"""
datasets_dir = 'C:/Users/Dhaval/Desktop/ASU/FSL/Project/code/'

def mnist(noTrSamples=1000, noTsSamples=100, \
                        digit_range=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], \
                        noTrPerClass=100, noTsPerClass=10):
    assert noTrSamples==noTrPerClass*len(digit_range), 'noTrSamples and noTrPerClass mismatch'
    assert noTsSamples==noTsPerClass*len(digit_range), 'noTrSamples and noTrPerClass mismatch'
    data_dir = os.path.join(datasets_dir, 'data/')
    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trData = loaded[16:].reshape((60000, 28*28)).astype(float)

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trLabels = loaded[8:].reshape((60000)).astype(float)

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    tsData = loaded[16:].reshape((10000, 28*28)).astype(float)

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    tsLabels = loaded[8:].reshape((10000)).astype(float)
    
    trData = np.copy(trData)
    tsData = np.copy(tsData)

    trData = trData/255.
    tsData = tsData/255.

    tsX = np.zeros((noTsSamples, 28*28))
    trX = np.zeros((noTrSamples, 28*28))
    tsY = np.zeros(noTsSamples)
    trY = np.zeros(noTrSamples)

    count = 0
    for ll in digit_range:
        # Train data
        idl = np.where(trLabels == ll)
        idl = idl[0][: noTrPerClass]
        idx = list(range(count*noTrPerClass, (count+1)*noTrPerClass))
        trX[idx, :] = trData[idl, :]
        trY[idx] = trLabels[idl]
        # Test data
        idl = np.where(tsLabels == ll)
        idl = idl[0][: noTsPerClass]
        idx = list(range(count*noTsPerClass, (count+1)*noTsPerClass))
        tsX[idx, :] = tsData[idl, :]
        tsY[idx] = tsLabels[idl]
        count += 1
    
    np.random.seed(1)
    test_idx = np.random.permutation(tsX.shape[0])
    tsX = tsX[test_idx,:]
    tsY = tsY[test_idx]

    trX = trX.T
    tsX = tsX.T
    trY = trY.reshape(1, -1)
    tsY = tsY.reshape(1, -1)
    return trX, trY, tsX, tsY

def load_data(noOfTrData, noOfTrDataPerClass, noOfValData, noOfValDataPerClass, noOfTestData, noOfTestDataPerClass, digits):
    trX, trY, tsX, tsY = mnist(noTrSamples=(noOfTrData + noOfValData),
                               noTsSamples=noOfTestData, digit_range=digits,
                               noTrPerClass=(noOfTrDataPerClass + noOfValDataPerClass), noTsPerClass=noOfTestDataPerClass)
    trX = trX.T
    tsX = tsX.T
    trY = trY[0]    
    
    dataPerClass = (noOfTrDataPerClass + noOfValDataPerClass)
    trXData = np.zeros((noOfTrData, 28*28))
    trYData = np.zeros((1, noOfTrData))
    valXData = np.zeros((noOfValData, 28*28))
    valYData = np.zeros((1, noOfValData))
    
    count = 0
    for i in range(0, len(digits)):
        
        idTr = list(range(count*noOfTrDataPerClass, (count+1)*noOfTrDataPerClass))
        idVal = list(range(count*noOfValDataPerClass, (count+1)*noOfValDataPerClass))
        #print(idTr)
        start = dataPerClass * count
        trXData[idTr, :] = trX[start:start + noOfTrDataPerClass, :]
        valXData[idVal, :] = trX[start + noOfTrDataPerClass:start + noOfTrDataPerClass + noOfValDataPerClass, :]
        trYData[0][idTr] = trY[start:start + noOfTrDataPerClass]
        valYData[0][idVal] = trY[start + noOfTrDataPerClass:start + noOfTrDataPerClass + noOfValDataPerClass]
        count += 1
        
    return trXData.T, trYData[0], valXData.T, valYData[0], tsX.T, tsY[0]


"""# Preprocess Data"""

label_title = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

noOfTrainingSample = 10000
noOfValidationSample = 1000
noOfTestSample = 5000
noOfTrainingSamplePerClass = int(noOfTrainingSample/10)
noOfValidationSamplePerClass = int(noOfValidationSample/10)
noOfTestSamplePerClass = int(noOfTestSample/10)

trX, trY, valX, valY, tsX, tsY = load_data(noOfTrainingSample, noOfTrainingSamplePerClass,
                                           noOfValidationSample, noOfValidationSamplePerClass,
                                           noOfTestSample, noOfTestSamplePerClass,
                                           [0,1,2,3,4,5,6,7,8,9])

def preprocess(data, test_data, val_data):
    total_features = data.shape[0]
    total_samples = data.shape[1]
    for feature in range(total_features):
        data[feature] -= np.mean(data[feature])
        test_data[feature] -= np.mean(data[feature])
        val_data[feature] -= np.mean(data[feature])
    return data.T, test_data.T, val_data.T

trX, tsX, valX = preprocess(trX, tsX, valX)

Y = np.zeros((10, trY.shape[0]))
for i in range(trY.shape[0]):
    Y[int(trY[i]), i] = 1
Y = np.insert(Y, 0, 1, axis = 0)



"""# Train"""

mm = ['nomomentum', 'poly', 'nag', 'rms', 'adam']
nets = []
dummyNet = NeuralNet(28*28,10)
dummyNet.add_fully_connected_layer(200, "relu")
dummyNet.add_fully_connected_layer(10, "relu")
dummyNet.build(output_activation="softmax", momentum='adam')

for i in range(5):
    net = NeuralNet(28*28,10)
    net.add_fully_connected_layer(200, "relu")
    net.add_fully_connected_layer(10, "relu")
    net.build(output_activation="softmax", momentum=mm[i])
    for i in range(4):
        net.layer[i].weights = np.copy(dummyNet.layer[i].weights)
    nets.append(net)

for i in range(5):
    nets[i].train(1001, trX, trY, Y, valX, valY, learning_rate=0.001, decay_rate=0, callFunc=None, callEvery=200, model=i)

def check_accuracy(data, labels, j):
    incorrect = 0
    for i in range(data.shape[0]):
        result = nets[j].predict(data[i])
        predicted_label = np.argmax(result)
        if labels[i] != predicted_label:
            incorrect = incorrect + 1
    print(mm[j], 1-((incorrect/1.0)/data.shape[0]))

for i in range(5):
    check_accuracy(tsX, tsY, i)
    times = np.array(nets[i].history['time'])
    print(np.sum(times))

	
"""# View"""
x = np.arange(0, len(nets[1].history['train_cost']), 1);
plt.plot(x, nets[0].history['validation_cost'], '--', label='No Momentum')
plt.plot(x, nets[1].history['validation_cost'], 'r', label='Poly')
plt.plot(x, nets[2].history['validation_cost'], 'y', label='NAG')
plt.plot(x, nets[3].history['validation_cost'], 'g', label='RMS')
plt.plot(x, nets[4].history['validation_cost'], 'b', label='Adam')
#plt.plot([250, 250], [0,2], '--')
axes = plt.gca()
#axes.set_xlim([60,80])
#axes.set_ylim([0,5])
plt.xlabel('Iterations')
plt.ylabel('Validation Loss')
plt.legend()
plt.show()

x = np.arange(0, len(net.history['train_cost']), 1);
plt.plot(x, nets[0].history['train_cost'], '--', label='No Momentum')
plt.plot(x, nets[1].history['train_cost'], 'r', label='Poly')
plt.plot(x, nets[2].history['train_cost'], 'y', label='NAG')
plt.plot(x, nets[3].history['train_cost'], 'g', label='RMS')
plt.plot(x, nets[4].history['train_cost'], 'b', label='Adam')
#plt.plot([250, 250], [0,2], '--')
axes = plt.gca()
#axes.set_xlim([60,80])
#axes.set_ylim([0,5])
plt.xlabel('Iterations')
plt.ylabel('Train Loss')
plt.legend()
plt.show()