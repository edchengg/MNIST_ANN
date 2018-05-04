import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
np.random.seed(0)


class ANN(object):
    def __init__(self, h, s):
        '''

        :param h: number of hidden layers
        :param s: number of hidden units in ALL layers
        '''
        self.layer_size = h
        self.hidden_units = s
        self.hidden_layers = None
        self.input_layer = None
        self.output_layer = None
        self.train_acc = []
        self.val_acc = []
        self.train_loss = []
        self.valid_loss = []
    def fit(self, X, y, alpha, t, X_valid, y_valid):
        '''

        :param X: Input matrix [n,m] e.g., 32000 * 784
        :param y: Label [n,1]   32000 * 1
        :param alpha: learning rate
        :param t: number of training epochs
        :param X_valid: Validation set
        :param y_valid: validation label
        :return: loss in each epoch
        '''
        # Initialize hidden layers, Input Layer, and Output Layer
        if self.hidden_layers is None:
            self.initialize_layer()
        if self.input_layer is None and self.output_layer is None:
            Input_layer = Layer(X=X[0], s=self.hidden_units,input=True)
            Output_layer = Layer(y_states=10, output=True)
            self.input_layer = Input_layer
            self.output_layer = Output_layer

        # Starts training
        LOSS = []
        for epoch in range(t):
            loss_epoch = 0
            stime = time()
            for i in range(len(X)):
                # Assign input to the input layer
                last_layer = self.input_layer
                last_layer.a = np.asarray(X[i]).reshape(-1,1)
                # Forward Propagation
                for hiddenlayer in self.hidden_layers:
                    hiddenlayer.input = np.dot(last_layer.weight.T, last_layer.a) + hiddenlayer.bias
                    #hiddenlayer.test_input1 = np.dot((last_layer.weight+1e-6).T, last_layer.a) + (hiddenlayer.bias+1e-6)
                    #hiddenlayer.test_input2 = np.dot((last_layer.weight-1e-6).T, last_layer.a) + (hiddenlayer.bias-1e-6)
                    # Activate the input
                    hiddenlayer.activation()
                    last_layer = hiddenlayer

                self.output_layer.input = np.dot(last_layer.weight.T, last_layer.a) + self.output_layer.bias
                #self.output_layer.test_input1 = np.dot((last_layer.weight+1e-6).T, last_layer.a) + (self.output_layer.bias+1e-6)
                #self.output_layer.test_input2 = np.dot((last_layer.weight-1e-6).T, last_layer.a) + (self.output_layer.bias-1e-6)
                self.output_layer.activation()

                # For label in y, make array with 0(False) and 1(True).
                # example: label = 1, array=[0,1,0,0,0,0,0,0,0,0]
                label = np.zeros((10,1))
                label[y[i]] = 1

                # Squared loss gradient
                loss = np.asarray(self.output_layer.a - label).reshape(-1,1)
                #pseudo_loss1 = np.asarray(self.output_layer.a1 - label).reshape(-1, 1)
                #pseudo_loss2 = np.asarray(self.output_layer.a2 - label).reshape(-1, 1)
                #loss1 = 0.5*np.sum(np.asarray(self.output_layer.a1 - label)**2)
                #loss2 = 0.5*np.sum(np.asarray(self.output_layer.a2 - label)**2)

                loss_epoch += np.sum(loss ** 2)

                # update output layer bias weights
                self.output_layer.backward(-loss)
                # Back Propagation
                last_layer = self.output_layer

                for hiddenlayer in reversed(self.hidden_layers):
                    hiddenlayer.backward(np.dot(hiddenlayer.weight, last_layer.error))
                    # Update current hidden layer weights
                    hiddenlayer.update(last_layer.error, alpha=alpha)
                    # Check gradient
                    #hiddenlayer.check_gradient(loss1,loss2, last_layer.error)

                    # Update last layer bias
                    last_layer.update_bias(alpha=alpha)
                    last_layer = hiddenlayer

                # Back propagation for input layer
                self.input_layer.backward(np.dot(self.input_layer.weight, last_layer.error))
                self.input_layer.update(last_layer.error, alpha=alpha)

                # Calculate average loss
                if i %5000 == 0 and i != 0:
                    print('Training Loss:', loss_epoch/i)


            print('epoch {}: training loss {}'.format(epoch + 1, loss_epoch/len(X)))
            LOSS.append(loss_epoch/len(X))

            # Evaluate on training set and validation set
            # % of correct classification
            acc_train,loss_train = self.evaluate(X, y)
            acc_valid, loss_valid = self.evaluate(X_valid, y_valid)
            self.train_acc.append(acc_train)
            self.val_acc.append(acc_valid)
            self.train_loss.append(loss_train)
            self.valid_loss.append(loss_valid)
            print('\nepoch %d, %.1f secs, lr = %.4f, train accuracy %.2f, val accuracy %.2f' % (epoch + 1,
                                                                              time() - stime,
                                                                              alpha,
                                                                              100 * acc_train,
                                                                              100 * acc_valid))
        return LOSS

    def predict(self, T):
        '''

        :param T: Input matrix size [k,m] e.g., 1000 * 784
        :return: Prediction Matrix size [k, n_class] e.g., 1000 * 10
        '''
        res = []
        for i in range(len(T)):
            last_layer = self.input_layer
            last_layer.a = np.asarray(T[i]).reshape(-1, 1)
            for hiddenlayer in self.hidden_layers:
                hiddenlayer.input = np.dot(last_layer.weight.T, last_layer.a) + hiddenlayer.bias
                hiddenlayer.activation()
                last_layer = hiddenlayer

            self.output_layer.input = np.dot(last_layer.weight.T, last_layer.a) + self.output_layer.bias
            self.output_layer.activation()

            res.append(self.output_layer.a.reshape(1,-1))

        return res

    def evaluate(self, X, y):
        '''

        :param X: Input matrix [n,m] e.g., 32000 * 784
        :param y: Label [n,1]   32000 * 1
        :return: Percentage of correctly classified examples
        '''
        correct = 0
        loss_total = 0
        for i in range(len(X)):
            last_layer = self.input_layer
            last_layer.a = np.asarray(X[i]).reshape(-1, 1)
            for hiddenlayer in self.hidden_layers:
                hiddenlayer.input = np.dot(last_layer.weight.T, last_layer.a) + hiddenlayer.bias
                hiddenlayer.activation()
                last_layer = hiddenlayer

            self.output_layer.input = np.dot(last_layer.weight.T, last_layer.a) + self.output_layer.bias
            self.output_layer.activation()

            label = np.zeros((10, 1))
            label[y[i]] = 1

            # Squared loss gradient
            loss = np.asarray(self.output_layer.a - label).reshape(-1, 1)
            loss_total += np.sum(loss ** 2)

            if np.argmax(self.output_layer.a) == y[i]:
                correct += 1
        return correct/len(X), loss_total/len(X)

    def print(self):
        print(self.input_layer.weight)
        for hiddenlayer in self.hidden_layers:
            print(hiddenlayer.weight)
        print(self.output_layer.weight)

    def loss_function(self, predict, y):
        # square error gradient
        return (predict - y)

    def initialize_layer(self, y_states=10):
        # Initialize hidden layer
        layer = []
        for i in range(self.layer_size - 1):
            layer.append(Layer(hidden=True,s=self.hidden_units, level=i))

        # Last hidden layer, specify number of classfication states in output layer
        layer.append(Layer(hidden=True, last_layer=True, s = self.hidden_units, y_states=y_states))

        self.hidden_layers = layer

class Layer(object):
    # Each layer contains a weight matrix, except the output layer
    # Each layer contains a bias vector, except the input layer
    # Backward, update weight, update bias are self contained functions
    def __init__(self, X=None, y_states=None, s=None, input=False, output=False, hidden=False, last_layer=False,level=0):
        '''

        :param X: Input matrix
        :param y_states: Number of classification states in output layer
        :param s: Number of hidden units in hidden layer
        :param input: Boolean if it is a input layer
        :param output: Boolean if it is an output layer
        :param hidden: Boolean if it is a hidden layer
        :param last_layer: Boolean if it is the last hidden layer before output layer
        :param level: Integer, level of the hidden layer
        '''
        if input:
            # Initialize Input Layer
            self.a = np.asarray(X).reshape(-1,1)
            self.weight = self.Xavier_init(len(self.a), s)
            self.input = self.a
            self.name = 'Input'


        elif output:
            # Initialize Output Layer
            self.input = np.zeros((y_states,1))
            self.bias = self.Xavier_init(y_states,1)
            self.name = 'Output'
            #self.test_input1 = np.zeros((y_states,1))
            #self.test_input2 = np.zeros((y_states, 1))

        elif hidden:
            # Initialize Hidden Layer
            if last_layer == False:
                self.weight = self.Xavier_init(s, s)
                self.bias = self.Xavier_init(s, 1)
                self.name = 'Hidden'+str(level + 1)

            else:
                # Initialize last hidden layer, number of weights determined by the output layer
                self.weight = self.Xavier_init(s, y_states)
                self.bias = self.Xavier_init(s, 1)
                self.name = 'Last_Hidden'


            self.input = np.zeros((s, 1))
            # error: a vector of errors by each hidden units
            self.error = np.zeros((s, 1))
            #self.test_input1 = np.zeros((s, 1))
            #self.test_input2 = np.zeros((s, 1))

        self.input_layer = input
        self.output_layer = output
        self.e = 1e-6

    def Xavier_init(self, pre, pos):
        '''

        :param pre: number of units in current layer
        :param pos: number of units in next layer
        :return: Weight matrix with Xavier Initialization weights
        '''
        # Xavier Initialization
        N = np.sqrt(6 / (pre + pos))
        W = np.random.uniform(low=-N, high=N, size=[pre, pos])
        return W

    def activation(self):
        # Activate input value --> a(input)
        self.a = self.sigmoid(self.input)
        #self.a1 = self.sigmoid(self.test_input1)
        #self.a2 = self.sigmoid(self.test_input2)

    def sigmoid(self, input):
        # Sigmoid function
        return 1 / (1 + np.exp(-input))

    def backward(self, loss):
        # Backward method find error
        self.error = self.gradient(self.input) * (loss)

    def gradient(self, input):
        # Gradient of sigmoid function
        return self.sigmoid(input) * (1 - self.sigmoid(input))

    def update(self, error, alpha):
        # Update weight matrix
        self.weight = self.weight + alpha * np.dot(self.a, error.T)

    def update_bias(self, alpha):
        # Update bias vector
        self.bias = self.bias + alpha * self.error

    def check_gradient(self, loss1, loss2, error):
        pseudo = (loss1 - loss2) / (2*1e-6)
        gradient = np.dot(self.a, error.T)
        #print(np.sum((gradient-pseudo.T)**2))




def plot_curve(train, val,loss=False):
    if loss:
        plt.xlabel("epochs")
        axis = np.arange(len(train)) + 1
        plt.ylabel("Loss")
        plt.plot(axis, [x for x in val])
        plt.plot(axis, [x for x in train])
        plt.legend(['Valid Loss', 'Train Loss'], loc='upper left')
        plt.savefig('loss')
        plt.close()
    else:
        plt.xlabel("epochs")
        axis = np.arange(len(train)) + 1
        plt.ylabel("Percentage")
        plt.plot(axis, [100 * x for x in val])
        plt.plot(axis, [100 * x for x in train])
        plt.legend(['Valid accuracy', 'Train accuracy'], loc='upper left')
        plt.savefig('acc')
        plt.close()


data = pd.read_csv('train.csv')
data_y = data['label'].values
data_X = data.iloc[:,1:].values/256

train_y, valid_y, test_y = data_y[:32000],data_y[32000:40000], data_y[40000:]
train_X, valid_X, test_X = data_X[:32000],data_X[32000:40000], data_X[40000:]

model = ANN(2, 256)


model.fit(train_X,train_y,1,2, valid_X, valid_y)
model.fit(train_X,train_y,0.5,2, valid_X, valid_y)
model.fit(train_X,train_y,0.2,2, valid_X, valid_y)
model.fit(train_X,train_y,0.1,2, valid_X, valid_y)

print("Test Accuracy: ",model.evaluate(test_X,test_y)[0])

res = model.predict(train_X)

plot_curve(model.train_acc, model.val_acc)
plot_curve(model.train_loss, model.valid_loss, loss=True)



