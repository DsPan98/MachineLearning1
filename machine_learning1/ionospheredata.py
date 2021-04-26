import numpy as np
import pandas as pd

from CleaningMethods import *



column = ['impulse 1-1', 'impulse 1-2', 'impulse 2-1','impulse 2-2', 'impulse 3-1', 'impulse 3-2', 'impulse 4-1', 'impulse 4-2', 'impulse 5-1', 'impulse 5-2', 'impulse 6-1','impulse 6-2','impulse 7-1','impulse 7-2','impulse 8-1','impulse 8-2','impulse 9-1','impulse 9-2','impulse 10-1','impulse 10-2','impulse 11-1', 'impulse 11-2', 'impulse 12-1','impulse 12-2', 'impulse 13-1', 'impulse 13-2', 'impulse 14-1', 'impulse 14-2', 'impulse 15-1', 'impulse 15-2', 'impulse 16-1','impulse 16-2','impulse 17-1','impulse 17-2','g/b'] 

IonosphereData = pd.read_csv('ionosphere.data', delimiter = ',', header = None, names = column)

IonoDataPanda = pd.DataFrame(data = IonosphereData)

pd.set_option('display.max_columns', 10)



"""
for x in column[:-1]: #loop through all but last g/b
    if IonoDataPanda[x].mean(axis = 0) == 0:
        y = x
        print(x)
# from this impulse 1-2 is broken

del IonoDataPanda[y]
"""

# ------------------------------------------------------------------------
#implemented methods case1

cm = CleaningMethods(data = IonoDataPanda, det_column = 'g/b', pos_value = ['g'])
cm.delete(all = True, use_missing = True, use_high_similarity = True, one_hot = True)
cm.normalize()
cm.fill_missing()
#cm.statistics(selected_feature = ['impulse 1-1', 'impulse 12-1'])






             
training = np.split(cm.data.to_numpy(),[200])[0] #200 is the splitting point
testing = np.split(cm.data.to_numpy(),[200])[1]

ClassIndex = training.shape[1] - 1 #get the index of the class attribute, the total num of features - 1

AllIndex = cm.data.iloc[:, ClassIndex].value_counts().index.tolist() #get all the unique values for class attribute

training[:, ClassIndex] = np.where(np.isin(training[:, ClassIndex], cm.pos_value), 1, training[:, ClassIndex])  #change all pos value to 1
training[:, ClassIndex] = np.where(np.isin(training[:, ClassIndex], list(set(AllIndex) - set(cm.pos_value))), 0, training[:, ClassIndex]) #change all non pos value to 0
                                                                                                                          
testing[:, ClassIndex] = np.where(np.isin(testing[:, ClassIndex], cm.pos_value), 1, testing[:, ClassIndex])
testing[:, ClassIndex] = np.where(np.isin(testing[:, ClassIndex], list(set(AllIndex) - set(cm.pos_value))), 0, testing[:, ClassIndex])

TrainingX = np.delete(training, training.shape[1]-1, axis = 1)
TrainingY = np.delete(training, range(training.shape[1] - 1), axis = 1)

TestingX = np.float64(np.delete(testing, testing.shape[1]-1, axis = 1))
TestingY = np.int32(np.delete(testing, range(testing.shape[1] - 1), axis = 1))


import numpy as np
from sklearn.model_selection import RepeatedKFold 
from sklearn.metrics import f1_score
class LogisticRegression:

    def __init__(self,learning_rate = 0.05):
        self.w = np.zeros((1,1))
        self.bias = np.zeros((1,1))
        self.learning_rate = learning_rate
        
    def sigmoid(self,Z):
        return 1/(1+np.exp(-Z))

    def loss(self,x,y):
        z = np.dot(x,self.w)
        return np.mean(y*np.log1p(np.exp(-z))+(1-y)*np.log1p(np.exp(z)))

    def gradient(self,x,y):
        N,D = x.shape
        yh = self.sigmoid(np.dot(x,self.w))
        grad = np.dot(x.T,yh-y)/N
        return grad
    
    def fit(self,x,y,learning_rate=0.01,eps=1e-2):
        m = len(y)
        self.w = np.float64(np.zeros((x.shape[1],1)))
        g = np.inf
        it = 0
        while np.linalg.norm(g) > eps and it < 100000:
            g = self.gradient(x,y)
            self.w = self.w - learning_rate*g
            it+= 1
            if it%1000 == 0:
                print (self.loss(x,y))
        print(it)


    def predict(self,x):
        preds = []
        Z = np.dot(x,self.w)
        for i in self.sigmoid(Z):
            if i>0.5:
                preds.append(1)
            else:
                preds.append(0)
        return preds

    def acc(self,x,y):
        print(f1_score(self.predict(x),y))

    def kfold(self,k,x,y):
        kf =RepeatedKFold(n_splits=k, n_repeats=1, random_state=None) 

        for train_index, test_index in kf.split(x):
            X_train, X_test = x[train_index], x[test_index] 
            y_train, y_test = y[train_index], y[test_index]
            self.fit(X_train,y_train,self.learning_rate)
            self.acc(X_test,y_test)
index = np.random.choice(200,200,replace=False)
lr = LogisticRegression()
x = np.float64(TrainingX[index])
y = np.int32(TrainingY[index])
