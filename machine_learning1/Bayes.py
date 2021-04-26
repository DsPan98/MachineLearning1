# Task Two - Abdullahi Elmi
import numpy as np
import pandas as pd
import sys
# import the math module  
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
np.set_printoptions(threshold=sys.maxsize)

# Importing the methods related to Task 1
import CleaningMethods
print("Cleaning Methods Imported")

import ionospheredata
ionTrainingX = np.float64(ionospheredata.trainingX)
ionTrainingY = np.float64(ionospheredata.trainingY)
ionTestingX = np.float64(ionospheredata.testingX)
ionTestingY = np.float64(ionospheredata.testingY)

import tictactoeData
tttTrainingX = tictactoeData.TrainingX
tttTrainingY = tictactoeData.TrainingY
tttTestingX = tictactoeData.TestingX
tttTestingY = tictactoeData.TestingY

# Starting With Naive Bayes
class NaiveBayes():
    # constructor
    def __init__(self, data_type, alpha=1.0):
        self.alpha = alpha
        self.separated_training={}
        self.data_type = data_type # datatype is going to tell the classifier which dataset was input, so that it knows
        # what bayes to use. 0 = adult dataset, 1 = ion dataset, 2 = credit dataset, 3 = tic tac toe dataset
    
    # Places the X dataset into a dictionary, separated by binary class (keys = 0,1)
    def class_separation(self, separated_training, X_training, y_training): # take your X & y training data sets and return a dictionary
            zero_start = True
            one_start = True
            # booleans to let us know that this is the first time we're encountering the class
            inst_count = y_training.shape[0]
            # Our number of instances/rows in our dataset
            for i in range(inst_count):
                # Runs for each instance
                X_copy = X_training[i,:].reshape(X_training[i,:].shape[0],1)
                # We create a copy of our X that adds another dimension (because numpy arrays are annoying)
                if y_training[i] == 0: # we place our current instance under the 0 key, if the corresponding y = 0
                    if zero_start == True: # in the case where this is the first zero we see, we create the key
                        separated_training[0] = X_copy
                        zero_start = False
                        # append doesn't work if this is the first element in the list
                    else:
                        separated_training[0] = np.append(separated_training[0], X_copy, axis=1)    
                        # just append the value if itsn't the first associated to the key
                elif y_training[i] == 1: # we place our current instance under the 1 key, if the corresponding y = 1
                    if one_start == True:
                        separated_training[1] = X_copy
                        one_start = False
                        # append doesn't work if this is the first element in the list
                    else: 
                        separated_training[1] = np.append(separated_training[1], X_copy, axis=1) 
                        # just append the value if itsn't the first associated to the key
            separated_training[0] = separated_training[0].T
            separated_training[1] = separated_training[1].T
            # just restructuring in order to have the rows in our dictionaries represent instances and not features
            return separated_training
    
    # Required fit function
    def fit(self, X, y):
        self.X_training = X
        self.y_training = y
        # the input datasets into the fit() function must be the training datasets
        
        self.separated_training[0] = np.array([[]])
        self.separated_training[1] = np.array([[]])
        # The class is always binary for this assignment, so 0 & 1 will always be the key in our separated dictionary
        self.separated_training = self.class_separation(self.separated_training, self.X_training, self.y_training)
        # Separate our training data by class
        
        # -----------------------------------Gaussian Training----------------------------------------
        self.mu_0 = np.mean(self.separated_training[0], axis=0)
        # array of means for class 0, each entry mapped to a feature
        self.mu_1 = np.mean(self.separated_training[1], axis=0)
        # array of means for class 0, each entry mapped to a feature
        self.sigma_0 = np.std(np.float64(self.separated_training[0]), axis=0)
        # array of standard deviations for class 0, each entry mapped to a feature
        self.sigma_1 = np.std(np.float64(self.separated_training[1]), axis=0)
        # array of standard deviations for class 1, each entry mapped to a features 
        # Now we calculate the necessary stats from the training data, the mean, and standard deviations for each class
        # So we get a mean and standard dev, for every feature, but separated based on class too
        
        # ------------------------------------Bernoulli Training---------------------------------------
        inst_count = X.shape[0] # the number of instances
        smoothing = 2 * self.alpha # smoothing function (not...really necessary)
        
        self.log_prior_0 = np.log(len(self.separated_training[0]) / inst_count)
        self.log_prior_1 = np.log(len(self.separated_training[1]) / inst_count)
        # the priors are calculated for each respective class
        
        self.likelihoods_0 = []
        self.likelihoods_1 = []
        for x in range(self.X_training.shape[1]):
            tp = (np.where((self.X_training[:, x]==1) &(self.y_training[:, 0]==1))[0].shape[0])/(np.where(self.y_training[:, 0]==1)[0].shape[0])
            fn = (np.where((self.X_training[:, x]==0) &(self.y_training[:, 0]==0))[0].shape[0])/(np.where(self.y_training[:, 0]==0)[0].shape[0])
            self.likelihoods_0.append(fn)
            self.likelihoods_1.append(tp)
        
    # The likelyhood function when using Gauss
    def gauss_likely(self, x, mu, sigma):
        numerator = np.exp(-(x - mu)**2/(2*sigma**2))
        denominator = (np.sqrt(2*np.pi)*sigma)
        return numerator*(1/denominator)
    
    # The likelyhood function when using Bernoulli
    def bern_likely(self, x, prior, likelihoods): 
        bern = np.log(likelihoods) * x + np.log(np.subtract(1, likelihoods)) * np.abs(x - 1) + prior 
        bern -= np.max(bern) #numerical stability
        posterior = np.exp(bern) # vector of size 2
        posterior /= np.sum(posterior) # normalize
        return posterior # posterior class probability
    
    # Returns the posterior probability for the ion dataset
    def posterior_prob_ion(self, X, X_training, mean_post, std_post):
            product = np.prod(self.gauss_likely(X, mean_post, std_post), axis=1)
            product = product*(X_training.shape[0]/self.X_training.shape[0])
            return product
    
    # Returns the posterior probability for the adult dataset
    def posterior_prob_adult(self, X, X_training, mean_post, std_post, prior, likelihoods):
            gauss_product = np.prod(self.gauss_likely(X, mean_post, std_post), axis=1)
            gauss_product = product*(X_training.shape[0]/self.X_training.shape[0])
            return gauss_product * bern_product
        
    # Returns the posterior probability for the credit dataset
    def posterior_prob_credit(self, X, X_training, mean_post, std_post, prior, likelihoods):
            gauss_product = np.prod(self.gauss_likely(X, mean_post, std_post), axis=1)
            gauss_product = product*(X_training.shape[0]/self.X_training.shape[0])
            return gauss_product * bern_product
        
    # Returns the posterior probability for the tic-tac-toe dataset
    def posterior_prob_ttt(self, X, X_training, prior, likelihoods):
            product = np.prod(self.bern_likely(X, prior, likelihoods), axis=1)
            #product = product*(X_training.shape[0]/self.X_training.shape[0])
            return product
            #print(self.bern_likely(X, prior, likelihoods).shape)
         
    def predict(self, X_test):
        if(self.data_type == 0): # mixed gauss & bernoulli
                return None# stuff we need to do if this is the adult data set
        elif(self.data_type == 1): # ionosphere case (purely gaussian)
            self.sigma_1[0] = self.sigma_1[0] + 0.00000001
            # first sigma_1 element is 0 for some reason at the start in the ion dataset
            zero_prob = self.posterior_prob_ion(np.float64(X_test), np.float64(self.separated_training[0]), np.float64(self.mu_0), self.sigma_0)
            one_prob = self.posterior_prob_ion(np.float64(X_test), np.float64(self.separated_training[1]), np.float64(self.mu_1), self.sigma_1)
            return 1*(one_prob > zero_prob)
        elif(self.data_type == 2): 
            return None# stuff we need to do if this is the credit dataset
        elif(self.data_type == 3): # tic-tac-toe case (pure bernoulli)
            zero_prob = self.posterior_prob_ttt(np.float64(X_test), np.float64(self.separated_training[0]), self.log_prior_0, self.likelihoods_0)
            one_prob = self.posterior_prob_ttt(np.float64(X_test), np.float64(self.separated_training[1]), self.log_prior_1, self.likelihoods_1)
            return 1*(one_prob > zero_prob)

nb_ion = NaiveBayes(1)
nb_ion.fit(ionTrainingX, ionTrainingY)
y_ion = nb_ion.predict(ionTestingX)
tp_ion = len([i for i in range(0, ionTestingY.shape[0]) if ionTestingY[i]==0 and y_ion[i]==0])
tn_ion = len([i for i in range(0, ionTestingY.shape[0]) if ionTestingY[i]==0 and y_ion[i]==1])
fp_ion = len([i for i in range(0, ionTestingY.shape[0]) if ionTestingY[i]==1 and y_ion[i]==0])
fn_ion = len([i for i in range(0, ionTestingY.shape[0]) if ionTestingY[i]==1 and y_ion[i]==1])
confusion_matrix_ion = np.array([[tp_ion, tn_ion],[fp_ion, fn_ion]])
print("Ionosphere Data, Confusion Matrix: \n", confusion_matrix_ion)

nb_ttt = NaiveBayes(3)
nb_ttt.fit(tttTrainingX, tttTrainingY)
y_ttt = nb.predict(tttTestingX)
#getting the confusion matrix
tp_ttt = len([i for i in range(0, tttTestingY.shape[0]) if tttTestingY[i]==0 and y_ttt[i]==0])
tn_ttt = len([i for i in range(0, tttTestingY.shape[0]) if tttTestingY[i]==0 and y_ttt[i]==1])
fp_ttt = len([i for i in range(0, tttTestingY.shape[0]) if tttTestingY[i]==1 and y_ttt[i]==0])
fn_ttt = len([i for i in range(0, tttTestingY.shape[0]) if tttTestingY[i]==1 and y_ttt[i]==1])
confusion_matrix_ttt = np.array([[tp_ttt, tn_ttt],[fp_ttt, fn_ttt]])
print("Tic Tac Toe Data, Confusion Matrix: \n", confusion_matrix_ttt)