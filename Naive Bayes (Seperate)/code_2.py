from mlxtend.plotting import plot_confusion_matrix
from utils import *
import numpy as np
import pandas as pd
import pprint
from itertools import chain
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

class Naive_Bayes:

    def _preprocess(self):
        '''
        Input: self.pos_train, self.neg_train, self.pos_test, self.neg_test
        Output: None

        Defines self.X_train, self.y_train, self.neg_test, self.pos_test, and self.word_dict
        '''
        for x in list(chain.from_iterable(self.pos_train)):
            if x in self.vocab:
                self.word_dict[x][0] += 1 #Stores all the words in from the review which are in vocab along with their frequency for all positive occurences
                self.N += 1
        
        for x in list(chain.from_iterable(self.neg_train)):
            if x in self.vocab:
                self.word_dict[x][1] += 1 #Stores all the words in from the review which are in vocab along with their frequency for all positive occurences
                self.N += 1
        
        pp = pprint.PrettyPrinter(indent=4, width = 5)
        self.X_train = self.pos_train+ self.neg_train
        self.y_train = [0] * len(self.pos_train) + [1]*len(self.neg_train)

        self.X_test = self.pos_test + self.neg_test
        self.y_test = [0]*len(self.pos_test) + [1]*len(self.neg_test)
    
    def __init__(self, pos_train, neg_train, pos_test, neg_test, vocab):
        self.N = 0
        self.pos_train = pos_train
        self.neg_train = neg_train
        self.pos_test = pos_test
        self.neg_test = neg_test
        self.vocab = vocab
        self.word_dict = {}
        for x in vocab:
            self.word_dict[x] = [0,0]    
        self._preprocess()

    def train(self, type = 1, alpha = 1):
        '''
        Inputs:
            type: set the type of formula to be use
            alpha: to be used when type is 2, will set the alpha value in the logarithmic function

        Output: None

        For different values of type, this function calculates the probability of occurence of each word in different context

        Note: Keep the value of type uniform across type and train
        '''
        V = len(self.vocab)

        if type == 2: #Log
            likelihood = {}
            prior = 0

            N_pos = N_neg = 0

            for x in self.word_dict:

                N_pos += self.word_dict[x][0]
                N_neg += self.word_dict[x][1]

            D = len(self.neg_train) + len(self.pos_train)

            D_pos = len(self.pos_train)

            D_neg = D - D_pos

            prior = np.log(D_pos) - np.log(D_neg)

            for w in self.word_dict:

                freq_pos = self.word_dict[w][0]
                freq_neg = self.word_dict[w][1]

                p_w_pos = (freq_pos+alpha) / (N_pos + alpha*V)
                p_w_neg = (freq_neg + alpha) / (N_neg + alpha*V)

                likelihood[w] = np.log(p_w_pos/p_w_neg)
            
            self.prior = prior
            self.likelihood = likelihood
        
        elif type ==1:
            

            count_0 = 0
            count_1 = 0
            for x in self.y_train:
                if x == 0:
                    count_0 += 1
                else:
                    count_1 += 1
            self.prob_0 = count_0/len(self.y_train)
            self.prob_1 = count_1/len(self.y_train)
            
            self.probabilities = self.word_dict
            for x in self.probabilities:
                self.probabilities[x][0] /= count_0
                self.probabilities[x][1] /= count_1

    def predict(self, word_l, type = 1):
        if type == 2:
            p = 0

            p += self.prior

            for w in word_l:

                if w in self.likelihood:
                    p += self.likelihood[w]
            return p
        
        elif type == 1:
            prod_0 = self.prob_0
            prod_1 = self.prob_1

            for w in word_l:
                prod_0 *= self.probabilities[w][0]
                prod_1 *= self.probabilities[w][1]
            
            if prod_1 > prod_0:
                return 1
            return 0

    def test(self, type = 1):
        '''
        Input: Internal class object (a word list)
        Output: y_hat: predicted value of y
        '''
        self.y_hats = []
        for r in self.X_test:
            word_l = []
            for w in r:
                if w in self.word_dict:
                    word_l.append(w)
            if type == 2:
                if self.predict(word_l, type = 2) > 0:
                    self.y_hats.append(0)
                else:
                    self.y_hats.append(1)
            elif type == 1:
                self.y_hats.append(self.predict(word_l))

        return self.y_hats    
    
    def performance(self, alpha = None):
        '''
        Calculates the performance of the alogrithm: accuracy_score, recall_score, f1_score, confusion_matrix
        Also prints the confusion matrix
        '''
        pp = pprint.PrettyPrinter(indent=4)
        #pp.pprint(confusion_matrix(self.y_test, self.y_hats))
        pp.pprint(f'Alpha {alpha}') if alpha is not None else print()
        pp.pprint(f'Accuracy Score: {accuracy_score(self.y_test, self.y_hats)}')
        pp.pprint(f'Recall Score:   {recall_score(self.y_test, self.y_hats)}')
        pp.pprint(f'F1 Score:       {f1_score(self.y_test, self.y_hats)}')

        fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_true=self.y_test, y_pred=self.y_hats), figsize=(6, 6), cmap=plt.cm.Blues)
        plt.xlabel('Predictions', fontsize=18)
        plt.ylabel('Actuals', fontsize=18)
        plt.title('Confusion Matrix', fontsize=18) if alpha is None else plt.title(f'Confusion Matrix alpha = {alpha}', fontsize=18) 
        plt.show()

        return accuracy_score(self.y_test, self.y_hats), recall_score(self.y_test, self.y_hats), f1_score(self.y_test, self.y_hats), confusion_matrix(self.y_test, self.y_hats)

# percentage_positive_instances_train = 0.2
# percentage_negative_instances_train = 0.2

# percentage_positive_instances_test  = 0.2
# percentage_negative_instances_test  = 0.2
	
# (pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train, percentage_negative_instances_train)
# (pos_test,  neg_test)         = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)
# n = Naive_Bayes(pos_train, neg_train, pos_test, neg_test, vocab)
# n.train(1)
# n.test(1)
# n.performance()

def question_1(n: Naive_Bayes):
    print("----------------------Question 1---------------------------------", end = "\n\n")

    #Standard without Log
    print("----------------------Standard Method---------------------------------", end = "\n\n")
    n.train(1)
    n.test(1)
    n.performance()
    print()
    #With Log
    print("----------------------Log Method---------------------------------", end = "\n\n")
    n.train(2)
    n.test(2)
    n.performance()
    print()
def question_2(n: Naive_Bayes):
    print("----------------------Question 2---------------------------------", end = "\n\n")

    alpha = 1e-4

    n.train(2)
    n.test(2)
    n.performance()
    print()

    score = []

    while alpha <= 1e3:
        n.train(2,alpha= alpha)
        n.test(2)
        a, _, _, _= n.performance(alpha)
        score.append(np.mean(a))
        print()
        alpha *= 10
    
    x = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3]

    plt.plot(x, score)
    plt.xlabel('Alpha Values')
    plt.ylabel('Accuracy Score')
    plt.xscale('log')
    plt.xticks(np.array(x))
    plt.show()

def question_3():
    print("----------------------Question 3---------------------------------", end = "\n\n")

    percent_positive_instance_train = 1
    percent_positive_instance_test = 1

    percent_negative_instance_train = 1
    percent_negative_instance_test = 1

    (pos_train, neg_train, vocab) = load_training_set(percent_positive_instance_train, percent_negative_instance_train)
    (pos_test,  neg_test)         = load_test_set(percent_positive_instance_test, percent_negative_instance_test)
    n = Naive_Bayes(pos_train, neg_train, pos_test, neg_test, vocab)

    n.train(type = 2, alpha = 0.01)
    n.test(type = 2)
    n.performance()
    print()

def question_4():
    print("----------------------Question 4---------------------------------", end = "\n\n")

    percent_positive_instance_train = 0.5
    percent_positive_instance_test = 1

    percent_negative_instance_train = 0.5
    percent_negative_instance_test = 1

    (pos_train, neg_train, vocab) = load_training_set(percent_positive_instance_train, percent_negative_instance_train)
    (pos_test,  neg_test)         = load_test_set(percent_positive_instance_test, percent_negative_instance_test)
    n = Naive_Bayes(pos_train, neg_train, pos_test, neg_test, vocab)

    n.train(type = 2, alpha = 0.01)
    n.test(type = 2)
    n.performance()
    print()

def question_6():
    print("----------------------Question 6---------------------------------", end = "\n\n")

    percent_positive_instance_train = 0.1
    percent_positive_instance_test = 1

    percent_negative_instance_train = 0.5
    percent_negative_instance_test = 1

    (pos_train, neg_train, vocab) = load_training_set(percent_positive_instance_train, percent_negative_instance_train)
    (pos_test,  neg_test)         = load_test_set(percent_positive_instance_test, percent_negative_instance_test)
    n = Naive_Bayes(pos_train, neg_train, pos_test, neg_test, vocab)

    n.train(type = 2, alpha = 0.01)
    n.test(type = 2)
    n.performance()
    print()

def main():

    percent_positive_instance_train = 0.2
    percent_positive_instance_test = 0.2

    percent_negative_instance_train = 0.2
    percent_negative_instance_test = 0.2

    (pos_train, neg_train, vocab) = load_training_set(percent_positive_instance_train, percent_negative_instance_train)
    (pos_test,  neg_test)         = load_test_set(percent_positive_instance_test, percent_negative_instance_test)
    n = Naive_Bayes(pos_train, neg_train, pos_test, neg_test, vocab)

    question_1(n)
    question_2(n)
    question_3()
    question_4()
    question_6()


if __name__ == "__main__":
    main()