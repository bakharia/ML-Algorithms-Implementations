'''
Name: Neural Network with back propagation
Author: Shubham Mishra
'''
from cmath import exp
from random import randint, random, seed, randrange
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


class NN:
    #Intitalizing functions
    def __init__(self, n_inputs, n_hidden, n_outputs, r) -> None:
        '''
        Intialize neural network
        - n_inputs: number of inputs
        - n_hidden: number of neurons in the hidden layer
        - n_outputs: number of outputs
        '''
        self.r = r
        try:
            hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
            self.network.append(hidden_layer)
            output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
            self.network.append(output_layer)
        except AttributeError:
            self.network = list()
            hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
            self.network.append(hidden_layer)
            output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
            self.network.append(output_layer)
        
    ### Forward Propagation Section ###
    def neuron_activation(self, weights, inputs):
        '''
        Activate one neuron given an input
        '''
        activation = weights[-1] #Assumption: bias is the last weight in the given list of weights

        # for i in range(len(weights)-1):
        #     activation += weights[i]* inputs[i]

        
        activation += np.dot(inputs,np.array(weights[:-1]).T)

        return activation
    
    def neuron_transfer(self, activation):
        '''
        Using sigmoid function to return the output of the neuron
        '''
        return 1.0/(1 + exp(-activation))
    
    def forward_propagate(self,  network, row):
        '''
        - network: The neural network
        - row: row of input data

        The neuron's output value is stored in the neuron with the name 'output'. 
        The outputs for a layer in an array named new_inputs is collected and that 
        becomes the array inputs which is used as inputs for the following layer.

        The function returns the outputs from the last layer also called the output layer.
        '''
        inputs = row

        for layer in network:
            new_input = []

            for neuron in layer:
                activation = self.neuron_activation(neuron['weights'], inputs)
                neuron['output'] = self.neuron_transfer(activation).real
                new_input.append(neuron['output'])
            inputs = new_input
        return inputs
    ### End of Section ###

    ### Backward Propagation Section ###

    def neuron_transfer_derivative(self, output):
        '''
        Returns the slope of the output value obtained from a neuron
        '''
        return output*(1 - output)
    
    def backward_propagate_error(self, network, expected):
        '''
        Backpropagate error and store in neurons
        '''

        for i in reversed(range(len(network))):
            layer = network[i]
            err = list()

            if i != len(network)-1:
                for j in range(len(layer)):
                    err_ = 0.0
                    for neuron in network[i+1]:
                        err_ += (neuron['weights'][j] * neuron['delta'])
                    err.append(err_)

            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    err.append(neuron['output'] - expected[j])
            
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = err[j] * self.neuron_transfer_derivative(neuron['output'])
    
    def v_backward_propagate_error(self, network, expected):
        '''
        Backpropagate error and store in neurons
        '''

        for i in reversed(range(len(network))):
            layer = network[i]
            err = list()

            if i != len(network)-1:
                weights = [n['weights'] for n in network[i+1]]
                delta = [n['delta'] for n in network[i+1]]

                err_ = np.dot(np.array(weights).T, delta)

                #print(err_)

                err.extend(err_)
                # for j in range(len(layer)):
                #     print(network[i+1])
                    
                #     #err_ = 0.0
                #     # for neuron in network[i+1]:
                #     #     err_ += (neuron['weights'][j] * neuron['delta'])
                #     err_ = np.dot(network[i+1]['delta'], np.array(network[i+1]['weights']).T)
                #     err.append(err_)

            else:
                output = np.array([n['output'] for n in layer])
                err.extend(np.subtract(output, expected))
            
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = np.dot(err[j], np.array(self.neuron_transfer_derivative(neuron['output'])).T)

    ### End of Section ###
    
    ### Training Section ###
    def update_weights(self, row, learning_rate):
        for i in range(len(self.network)):
            inputs = row[:-1]
            
            if i != 0:
                inputs = [neuron['output'] for neuron in self.network[i - 1]]
            for neuron in self.network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] -= learning_rate * neuron['delta'] * inputs[j]
                    neuron['weights'][-1] -= learning_rate * neuron['delta']

    def train_network(self, train, target:int, learning_rate, n_epochs, n_outputs):
        self.target = target
        self.create_index(train)

        for epoch in range(n_epochs):
            sum_err = 0

            for row in train:
                outputs = self.forward_propagate(self.network, row)
                expected = [0 for i in range(n_outputs)]
                #print(outputs)
                # print(row[target])
                expected[self.get_index(row[self.target])] = 1
                
                
                sum_err += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])

                self.backward_propagate_error(self.network, expected)

                self.update_weights(row, learning_rate)

            print(f'>epoch={epoch}, lrate={learning_rate}, error={sum_err}')


    ### End of Section ###
    
    ### Prediction Section ###
    def predict(self, row):
        '''
        # Make a prediction with a network
        '''
        outputs = self.forward_propagate(self.network, row)
        return list(self.index.keys())[outputs.index(max(outputs))]
        #return outputs.index(max(outputs))
    ### End of Section ###

    #Get Functions
    def get_network(self) -> list:
        '''
        Get the intialized network
        '''
        try:
            return self.network
        except AttributeError:
            print("Network Not initialised, returning empty list")
            return list()
    
    def create_index(self, data):
        i = 0
        self.index = {}
        values = list(set([row[self.target] for row in data]))

        for v in values:
            self.index[v] = i
            i += 1
        
        print(self.index)

    def get_index(self, value)->int:
        return self.index[value]

 
def stratified_cross_validation(dataset: pd.DataFrame, target:int, n_folds = 3):
    '''
    Returns stratified k-fold dataset
    '''
    target_class, target_count = np.unique(dataset.iloc[:, target], return_counts=True)

    dataset_copy = dataset.sort_values(by=dataset.columns[0]).copy()

    prob_class = target_count/np.sum(target_count)
    # print(dataset_copy)

    folds = []
    sub_dataset = []

    fold_size = len(dataset) // n_folds

    target_count = (prob_class * fold_size)//1

    

    for i in range(len(target_class)):
        temp = dataset_copy.loc[dataset_copy[dataset_copy.columns[target]] == target_class[i]].copy()
        sub_dataset.append(temp)
    
    for i in range(n_folds):
        fold = 0
        for j in range(len(target_class)):
            
            selected_rows = list(np.random.choice(sub_dataset[j].index, int(target_count[j])))
            
            temp = sub_dataset[j].loc[selected_rows]
            
            sub_dataset[j] = sub_dataset[j].drop(index = list(selected_rows))
            
            #print(len(temp), len(sub_dataset[j]))
            
            if isinstance(fold, int):
                fold = temp
            else:
                fold = pd.concat([fold, temp], ignore_index=True)

        folds.append(fold)

    return folds

def accuracy_metric(true, predict, data):
    key = list(set([row[-1] for row in data]))

    values = list(set(true))

    print(key, values)

    dict = {}
    for i in range(len(key)):
        dict[values[i]] = key[i]

    true = [dict[x] for x in true]
    predict = [dict[x] for x in predict]

    return accuracy_score(true, predict), f1_score(true, predict, average="macro")

     

def main():
    seed(42)
    #data = pd.read_csv('../files/datasets/hw3_cancer.csv', sep = '\t')
    #data = pd.read_csv('../dataset/hw4_house_votes_84.csv')
    data = pd.read_csv('../files/datasets/hw3_wine.csv', sep = '\t')
    #data = pd.read_csv('..\dataset\cmc.csv', header= None) #read your file
    
    k_fold = 10 #set the amount of cross folds
    target = -1 #col no of the target variable
    n_hidden_layers = 7 #select number of hidden layers
    learning_rate = 0.5 #select the learning rate
    epochs = 50 #number of times to be trained 
    lmbda = 0.1 #regularization param

    #Dont change anything beyond this point

    att_ = []
    target_ = data.columns[target:target+1]
    for i in range(len(data.columns)):
        if i != target:
            att_.append(data.columns[i])
    
    data = data[att_ + list(target_)]

    data_split = stratified_cross_validation(data, -1, k_fold)
    
    acc_score = 0
    f_score = 0

    for i in range(k_fold):
        val = data_split[i]
        
        # print(len(val))
        train = 0
        acc_score_k = []
        f_score_k = []
        for x in [k for k in range(k_fold) if k != i]:


            train = pd.concat([train, data_split[x]], ignore_index=True) if not isinstance(train, int) else data_split[x]

            print(f"Starting batch process \n Mini Batch size {len(train)}")

            scaler = MinMaxScaler().fit(train)
            train_set = scaler.transform(train)
            
            val_set = scaler.transform(val)
            
            # print(train)
            # print(val)

            n_inputs = len(train_set[0]-1)
            n_outputs = len(set([row[-1]  for row in train_set]))
        

            network = NN(n_inputs, n_hidden_layers, n_outputs, lmbda)
            network.train_network(train_set, -1, learning_rate, epochs, n_outputs)

            # print(len(network.get_network()))

            predictions = []
            actual = []
            for x in val_set:
                predictions.append(network.predict(x))
                actual.append(x[-1])
            
            #print(actual, predictions)
            acc, f1 = accuracy_metric(actual, predictions, data)
            acc_score_k.append(acc)
            f_score_k.append(f1)
            #print(acc, f_score)
            print("Model Performance: \n Accuarcy: {:.2f} \t F1 Score: {:.2f} for batch size {:d}".format(acc,f1, len(train)))

        acc_score = np.array(acc_score_k) if isinstance(acc_score, int) else (acc_score + np.array(acc_score_k))
        f_score = np.array(f_score_k) if isinstance(f_score, int) else (f_score + np.array(f_score_k))
    
    acc_score /= k_fold
    f_score /= k_fold

    print(acc_score, f_score)

    with plt.style.context('seaborn-darkgrid'):
        plt.figure()
        plt.plot(acc_score, label = 'Accuracy')
        plt.plot(f_score, label = 'F1 Score')
        plt.xticks([i for i in range(k_fold-1)],[f'{i}/{k_fold}' for i in range(1,k_fold)])
        plt.xlabel('Batch Used')
        plt.legend(['Accuracy', 'F1 Score'])
        plt.show()
        plt.close()
    # for layer in network.get_network():
    #     print(layer)


# def test():
    
#     nn = NN(1,2,1)

#     print(nn.get_network())

if __name__ == '__main__':

    main()

