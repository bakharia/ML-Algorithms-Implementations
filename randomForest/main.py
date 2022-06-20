import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from random_forest import random_forest_algorithm, random_forest_predictions
from helper_functions import calculate_accuracy,train_test_split, cross_validation,calculate_confusionMatrix, calculate_f1score, calculate_precision, calculate_recall
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

if __name__ == "__main__":

    df = pd.read_csv('../datasets/cmc.csv', sep = ',')
    #df = df.iloc[:,-1::-1]
    train_df, test_df = train_test_split(df, 0.3)

    #cross_validation(df,3)

    #forest = random_forest_algorithm(train_df, n_trees=4, n_bootstrap=800, n_features=2, dt_max_depth=4)
    # predictions = random_forest_predictions(test_df, forest)
    # accuracy = calculate_accuracy(predictions, test_df.iloc[:,-1])

    # print("Accuracy = {}".format(accuracy))

    score_f = []
    precision_score_f = []
    recall_score_f = []
    f1_score_f = []
    #confusion_matrix_f = []
    folds = cross_validation(train_df, n_folds=10)
    for n_trees in [1,5,10,20,30,40,50]:
        score = []
        precision_score = []
        recall_score = []
        f1_score = []
        confusion_matrix = []
        
        f = 0
        for fold in folds:
            train_df = folds.copy()
            train_df.pop(f)
            train_df = pd.concat(train_df)
            # forest = random_forest_algorithm(train_df, n_trees=n_trees, n_bootstrap=800, n_features=2, dt_max_depth=4, option = 1)
            # predictions = np.array(random_forest_predictions(fold,forest), dtype=np.int8)
            rf = RandomForestClassifier(n_estimators=n_trees, criterion="entropy", random_state=0)
            rf.fit(np.array(train_df.iloc[:,:-1], dtype=np.int64), np.array(train_df.iloc[:,-1],dtype=np.int64))
            predictions = rf.predict(np.array(fold.iloc[:,:-1],dtype=np.int64))
            labels = np.array(fold.iloc[:,-1], dtype=np.int8)

            score.append(calculate_accuracy(predictions, labels))
            precision_score.append(calculate_precision(predictions, labels))
            recall_score.append(calculate_recall(predictions, labels))
            f1_score.append(calculate_f1score(predictions, labels))
            #confusion_matrix.append(calculate_confusionMatrix(predictions, labels))
            f += 1
        score_f.append(np.mean(score))
        precision_score_f.append(np.mean(precision_score))
        recall_score_f.append(np.mean(recall_score))
        f1_score_f.append(np.mean(f1_score))
    print(score_f, np.mean(score))
    print(precision_score_f, np.mean(precision_score))
    print(recall_score_f, np.mean(recall_score))
    print(f1_score_f, np.mean(f1_score))
    #print(confusion_matrix)
    plt.figure()
    sns.lineplot(x = [1,5,10,20,30,40,50], y = score_f).set(xlabel = 'n_trees', ylabel = 'Score', title = 'Accuracy Score vs n_tree')
    plt.show()
    plt.close()
    plt.figure()
    sns.lineplot(x = [1,5,10,20,30,40,50], y = precision_score_f).set(xlabel = 'n_trees', ylabel = 'Score', title = 'Precision Score vs n_tree')
    plt.show()
    plt.close()
    plt.figure()
    sns.lineplot(x = [1,5,10,20,30,40,50], y = recall_score_f).set(xlabel = 'n_trees', ylabel = 'Score', title = 'Recall Score vs n_tree')
    plt.show()
    plt.close()
    plt.figure()
    sns.lineplot(x = [1,5,10,20,30,40,50], y = f1_score_f).set(xlabel = 'n_trees', ylabel = 'Score', title = 'F1 Score vs n_tree')
    plt.show()
    plt.close()

