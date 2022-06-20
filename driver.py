######### FINAL PROJECT ############
from sklearn import datasets
from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
import random
import json

#Importing Decision Tree
from decisionTree.decision_tree import DecisionTree 
#Importing Neural Network
from neuralNetwork.nn import NN, accuracy_metric as nn_accuracy_metric
#Importing K-NN
from knn.knn import knn, accuracy
#Importing Random Forest
from randomForest.random_forest import random_forest_algorithm, random_forest_predictions
from randomForest.helper_functions import calculate_accuracy, cross_validation,calculate_confusionMatrix, calculate_f1score, calculate_precision, calculate_recall

def call_knn(train: pd.DataFrame, 
            val: pd.DataFrame, 
            k_fold:int):
    '''
    Author: Ankit Kumar

    Returns the predicted values
    '''
    print(f"########### Implementing k-nn  {k_fold} ###########")
    if k_fold%2 == 0:
        k_fold += 1
        print("Changed k to odd", k_fold)
    
    X_train = train.iloc[:,:-1].to_numpy()
    y_train = train.iloc[:,-1].to_numpy()

    X_test = val.iloc[:,:-1].to_numpy()
    y_test = val.iloc[:,-1].to_numpy()

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    knn_cl = knn(k = k_fold)

    knn_cl.fit(X_train, y_train)

    pred_train = knn_cl.predict(X_train)

    pred_test = knn_cl.predict(X_test)

    return pred_test, pred_train, y_test, y_train
    
    


def call_decisionTree(train, val, min_split = 2, max_depth = 5):
    '''
    Author: Shubham Mishra

    Returns the predicted values
    '''
    print("########### Implementing Decision Tree  ###########")
    
    X_train = train.iloc[:, :-1].to_numpy()
    y_train = train.iloc[:, -1].to_numpy()

    X_test = val.iloc[:, :-1].to_numpy()
    y_test = val.iloc[:, -1].to_numpy()

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    model = DecisionTree(min_samples_split=min_split, max_depth= max_depth)
    model.fit(X_train, y_train)

    test_preds = model.predict(X_test)
    test_scores = accuracy_score(y_test, test_preds)

    train_preds = model.predict(X_train)
    train_scores = accuracy_score(y_train, train_preds)

    return test_preds, train_preds, y_test, y_train

def call_randomForest(train: pd.DataFrame, val: pd.DataFrame, n_trees:int):
    '''
    Author: Shubham Mishra

    Returns the predicted values
    '''
    print("########### Implementing Random Forest  ###########")
    # forest = random_forest_algorithm(train_df, n_trees=n_trees, n_bootstrap=800, n_features=2, dt_max_depth=4, option = 1)
    # predictions = np.array(random_forest_predictions(fold,forest), dtype=np.int8)

    train.columns = [f'{x}' for x in train.columns]
    val.columns = [f'{x}' for x in val.columns]

    train_col = train.columns
    val_col = val.columns

    sc = StandardScaler()
    train = pd.DataFrame(sc.fit_transform(train), columns=train_col)
    val = pd.DataFrame(sc.transform(val), columns=val_col)

    rf = random_forest_algorithm(train, n_trees, 800, 4, 4, 2)
    
    pred_train = np.array(random_forest_predictions(train, rf))
    pred_val = np.array(random_forest_predictions(val, rf))

    return pred_val, pred_train, val.iloc[:,-1].values, train.iloc[:,-1].values



def call_neuralNetwork(train:pd.DataFrame,
                    val:pd.DataFrame,
                    target:int,
                    n_hidden_layers: int,
                    learning_rate: float,
                    epochs: np.uint):
    '''
    Author: Shubham Mishra & Ankit Kumar

    Returns the predicted values
    '''
    scaler = MinMaxScaler()
    
    train_set = scaler.fit_transform(train)
            
    val_set = scaler.transform(val)

    n_inputs = len(train_set[0])
    n_outputs = len(set([row[-1]  for row in train_set]))
        

    network = NN(n_inputs, n_hidden_layers, n_outputs, 1)
    network.train_network(train_set, -1, learning_rate, epochs, n_outputs)

    pred_test = []
    actual_test = []
    
    for x in val_set:
        pred_test.append(network.predict(x))
        actual_test.append(x[-1])
    
    pred_train = []
    actual_train = []
    for x in train_set:
        pred_train.append(network.predict(x))
        actual_train.append(x[-1])

    return pred_test, pred_train, actual_test, actual_train

    
            
    

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

def train_test_split_custom(X, y, test_size = 0.15):

    if isinstance(test_size, float):
        test_size = round(test_size * len(X))

    indices = range(len(X))
    test_indices = random.sample(population=indices, k=test_size)

    test_X = X[test_indices]
    train_X = np.delete(X, test_indices, 0)
    test_y = y[test_indices]
    train_y = np.delete(y, test_indices, 0)

    print(len(test_y), len(train_y))
    
    return train_X, test_X, train_y, test_y


def ensemble(results):
    '''
    Using majority voting to select the best result from multiple outputs
    '''

    final_result = []

    results = np.array(results).T
    print(results)
    for i in range(results.shape[0]):
            final_result.append(np.bincount(results[i]).argmax())

    return final_result

def driver_function():

    k_fold = 3

    df = pd.read_csv('dataset\\titanic.csv')
    df = df.drop(columns= ['Name'])
    df = pd.get_dummies(df, columns = ['Sex'])
    df = df[['Survived', 'Pclass', 'Age', 'Siblings/Spouses Aboard',
        'Parents/Children Aboard', 'Sex_female', 'Sex_male', 'Fare']]

    

    df.iloc[:, -1] = df.iloc[:,-1].apply(lambda x: 0 if x > 70 else 1 if x > 40 else 2)
    df.iloc[:,  2] = df.iloc[:, 2].apply(lambda x: int(x))

    #dataset = pd.read_csv('dataset\parkinsons.csv')

    print(df)

    data_split = stratified_cross_validation(pd.DataFrame(df), -1, k_fold)

    dict = {
        'knn': {
            'pred_test': [],
            'pred_train': [],
            'actual_test': [],
            'actual_train': [],
            'accuracy_score': [0]*4,
            'params': [],
            'f1_score': [0]*4
        },
        'decision_tree': {
            'pred_test': [],
            'pred_train': [],
            'actual_test': [],
            'actual_train': [],
            'accuracy_score': [0]*4,
            'params': [],
            'f1_score': [0]*4
        },
        'random_forest': {
            'pred_test': [],
            'pred_train': [],
            'actual_test': [],
            'actual_train': [],
            'accuracy_score': [0]*4,'params': [],
            'f1_score': [0]*4
        },
        'nn': {
            'pred_test': [],
            'pred_train': [],
            'actual_test': [],
            'actual_train': [],
            'accuracy_score': [0]*3,'params': [],
            'f1_score': [0]*3
        }
    }

    val_idx = 0
    for i in range(k_fold):
        
        train_data = 0

        for i in range(len(data_split)):

            if i != val_idx:

                if isinstance(train_data, int):
                    train_data = data_split[i]
                else:
                    train_data = pd.concat([train_data, data_split[i]])

        
        val_data = data_split[val_idx]

        train_data_knn = train_data.copy()
        val_data_knn = val_data.copy()

        train_data_knn.iloc[:,-1] = train_data_knn.iloc[:,-1].apply(lambda x: int(x))
        val_data_knn.iloc[:,-1] = val_data_knn.iloc[:,-1].apply(lambda x: int(x))
        #print(call_decisionTree(train_data, val_data))
        #print(call_knn(train_data, val_data, 5))
        #print(call_randomForest(train_data, val_data, 8))
        #print(call_neuralNetwork(train_data, val_data, -1, 8, 0.5, 50))

        ### Using k-NN
        size = len(train_data)
        print("Size of the dataset is", size)
        k = int(np.sqrt(size) // 1)

        if k % 2:
            k = [k-4, k-2, k+2,k+ 4]
        else:
            k = [k-3, k-1, k+1, k+3]
        
        i = 0
        for kv in k:
            pred_test, pred_train, y_test, y_train = call_knn(train_data_knn, val_data_knn, kv)

            #print(accuracy_score(y_test, pred_test), accuracy_score(y_train, pred_train))
            dict['knn']['pred_test'].append([pred_test])
            dict['knn']['pred_train'].append([pred_train])
            dict['knn']['actual_test'].append([y_test])
            dict['knn']['actual_train'].append([y_train])
            dict['knn']['accuracy_score'][i] += accuracy_score(y_test, pred_test)
            dict['knn']['f1_score'][i] += f1_score(y_test, pred_test, average = 'macro')
            dict['knn']['params'].append(kv)
            i += 1

        
        #### Using Decision Tree

        params = [(3,5), (4,5), (5,5)]

        i = 0
        for s, d in params:
            pred_test, pred_train, y_test, y_train = call_decisionTree(train_data, val_data, s, max_depth=d)

            #print(accuracy_score(y_test, pred_test), accuracy_score(y_train, pred_train))

            dict['decision_tree']['pred_test'].append([pred_test])
            dict['decision_tree']['pred_train'].append([pred_train])
            dict['decision_tree']['actual_test'].append([y_test])
            dict['decision_tree']['actual_train'].append([y_train])
            dict['decision_tree']['accuracy_score'][i] += accuracy_score(y_test, pred_test)
            dict['decision_tree']['f1_score'][i] += f1_score(y_test, pred_test, average = 'macro')
            dict['decision_tree']['params'].append((s,d))
            i += 1
        
        #### Using Random Forest

            # # n_features = int(np.sqrt(train_data.shape[1]))
            
            # # params = [n_features]
            
            # # i = 0
            # # for n in params:
            # #     pred_test, pred_train, y_test, y_train = call_randomForest(train_data, val_data, n)

            # #     #print(accuracy_score(y_test, pred_test), accuracy_score(y_train, pred_train))

            # #     dict['random_forest']['pred_test'].append([pred_test])
            # #     dict['random_forest']['pred_train'].append([pred_train])
            # #     dict['random_forest']['actual_test'].append([y_test])
            # #     dict['random_forest']['actual_train'].append([y_train])
            # #     dict['random_forest']['accuracy_score'][i] += accuracy_score(y_test, pred_test)
            # #     dict['random_forest']['f1_score'][i] += f1_score(y_test, pred_test, average = 'macro')
            # #     dict['random_forest']['params'].append(n)

            # #     i += 1

        # i = 0
        # n_features = train_data.shape[0]
        # print(n_features)
        # params = [(n_features - 1, 0.8), (n_features - 3, 0.5), (n_features - 4, 0.5)]

        # for h_la, lr in params:
        #     pred_test, pred_train, y_test, y_train = call_neuralNetwork(train_data, val_data, -1, h_la, lr, 50)

        #     #print(accuracy_score(y_test, pred_test), accuracy_score(y_train, pred_train))

        #     dict['nn']['pred_test'].append([pred_test])
        #     dict['nn']['pred_train'].append([pred_train])
        #     dict['nn']['actual_test'].append([y_test])
        #     dict['nn']['actual_train'].append([y_train])
        #     dict['nn']['accuracy_score'][i] += accuracy_score(y_test, pred_test)
        #     dict['nn']['f1_score'][i] += f1_score(y_test, pred_test, average = 'macro')

        #     i += 1

    # print(X_train, len(X_train))
    # print(X_test, len(X_test))
    # print(y_train, len(y_train))
    # print(y_test, len(y_test))

    #print(call_decisionTree(X_train, X_test, y_train, y_test))
    for k in dict:
        print(k)
        dict[k]['accuracy_score'] = np.array(dict[k]['accuracy_score']) / k_fold
        dict[k]['f1_score'] = np.array(dict[k]['f1_score'])/k_fold
        print(dict[k])
        if k == 'knn':
            with plt.style.context('seaborn-darkgrid'):
                plt.figure()
                plt.plot(list(np.unique(dict[k]['params'])),dict[k]['accuracy_score'], label = 'accuracy')
                plt.plot(list(np.unique(dict[k]['params'])),dict[k]['f1_score'], label = 'f1')
                plt.xticks(list(np.unique(dict[k]['params'])))
                plt.legend(['Accuracy', 'F1 Score'])
                plt.xlabel('Params')
                plt.ylabel('Score')
                plt.savefig(f'results/titanic_{k}_params.png')
                plt.show()
                plt.close()
        if k == 'decision_tree':
            with plt.style.context('seaborn-darkgrid'):
                plt.figure()
                # print(list(np.unique(dict[k]['params'])))
                print([f'({x},{y})' for x,y in dict[k]['params'][:4]])
                plt.plot([f'({x},{y})' for x,y in dict[k]['params'][:4]],dict[k]['accuracy_score'], label = 'accuracy')
                plt.plot([f'({x},{y})' for x,y in dict[k]['params'][:4]],dict[k]['f1_score'], label = 'f1')
                plt.xticks([f'({x},{y})' for x,y in dict[k]['params'][:4]])
                plt.legend(['Accuracy', 'F1 Score'])
                plt.xlabel('Params (min_split, max_depth)')
                plt.ylabel('Score')
                plt.savefig(f'results/titanic_{k}_params.png')
                plt.show()
                plt.close()
        # if k == 'random_forest':
        #     with plt.style.context('seaborn-darkgrid'):
        #         plt.figure()
        #         plt.plot(list(np.unique(dict[k]['params'])),dict[k]['accuracy_score'], label = 'accuracy')
        #         plt.plot(list(np.unique(dict[k]['params'])),dict[k]['f1_score'], label = 'f1')
        #         plt.xticks(list(np.unique(dict[k]['params'])))
        #         plt.legend(['Accuracy', 'F1 Score'])
        #         plt.xlabel('Params (n_trees)')
        #         plt.ylabel('Score')
        #         plt.savefig(f'results/d_{k}_params.png')
        #         plt.show()
        #         plt.close()
        


def test():
    print(ensemble(results=[
        [1, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 1, 1],
        [0, 1, 1, 0, 0, 0]
    ]))
    
if __name__ == '__main__':
    driver_function()