import pandas as pd

from CleaningMethods import *

column = ['top-left-square', 'top-middle-square', 'top-right-square', 'middle-left-square', 'middle-middle-square', 'middle-right-square', 'bottom-left-square', 'bottom-middle-square', 'bottom-right-square', 'Class']


tictactoeData = pd.read_csv('tic-tac-toe.data', delimiter = ',', header = None, names = column)
tictactoePanda = pd.DataFrame(data = tictactoeData, columns = column)

pd.set_option('display.max_columns', tictactoePanda.shape[1])

cm = CleaningMethods(data = tictactoePanda, det_column = 'Class', pos_value = ['positive'])

cm.delete(use_high_similarity = True, use_collinear = True, one_hot = True)

cm.normalize()

cm.fill_missing()

#cm.statistics(selected_feature = ['top-left-square', 'top-middle-square'])

selectTraining = np.sort(np.random.choice(958, 500,replace = False))
selectTesting = list(set(range(958)) - set(selectTraining))

XPanda1 = cm.data.drop(cm.data.columns[cm.data.shape[1] - 1], axis = 1)
YPanda = cm.data.iloc[:, cm.data.shape[1] - 1]
XPanda2 = pd.get_dummies(XPanda1) #unsplit x attributes, pandas form

XSet = XPanda2.to_numpy() #unsplit x attributes, np form
YClass = YPanda.to_numpy()

AllIndex = YPanda.value_counts().index.tolist()

YClass = np.where(np.isin(YClass, cm.pos_value), 1, YClass)
YClass = np.where(np.isin(YClass, list(set(AllIndex) - set(cm.pos_value))), 0, YClass)

TrainingX = XSet[selectTraining]
TestingX = np.float64(XSet[selectTesting])

TrainingY = YClass[selectTraining]
TrainingY.shape = [500, 1]
TestingY = np.float64(YClass[selectTesting])
TestingY.shape = [458,1]

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

    def loss(self,y,y_hat):
        return -np.mean(y*np.log(y_hat)+(1-y)*np.log(1-y_hat))

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
        while np.linalg.norm(g) > eps and it<100000:
            g = self.gradient(x,y)
            self.w = self.w - learning_rate*g
            yh = self.sigmoid(np.dot(x,self.w))
            it+= 1
            if it%1000 == 0:
                print (self.loss(y,yh))
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
index = np.random.choice(500,100,replace=False)
lr = LogisticRegression()
x = np.float64(TrainingX[index])
y = np.int32(TrainingY[index])

