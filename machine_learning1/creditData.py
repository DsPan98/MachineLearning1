import numpy as np
import pandas as pd


from CleaningMethods import *




column = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16']

creditData = pd.read_csv('crx.data', delimiter = ',', header = None, names = column)

creditPanda = pd.DataFrame(data = creditData, columns = column)          

pd.set_option('display.max_columns', creditPanda.shape[1])


# ------------------------------------------------------------------------
#implemented methods case1 
cm = CleaningMethods(data = creditPanda, det_column ='A16', pos_value = ['+'])


cm.delete(use_high_similarity = True, use_ordinal = True, use_low_importance = True, one_hot = True )
cm.normalize()
cm.fill_missing(missing_indication = '?')
#cm.statistics(selected_feature = ['A2', 'A3'])


XPanda1 = cm.data.drop(cm.data.columns[cm.data.shape[1] - 1], axis = 1)
YPanda = cm.data.iloc[:, cm.data.shape[1] - 1]

XPanda2 = pd.get_dummies(XPanda1) #unsplit x attributes, pandas form

One_Hot_Index = XPanda2.columns.tolist()
#continuousFeature = list(set(One_Hot_Index).intersection(column))
#binaryFeature = list(set(One_Hot_Index) - set(column))

XcontinuousPanda = XPanda2.loc[:, set(One_Hot_Index).intersection(column)]
XbinaryPanda = XPanda2.loc[:, list(set(One_Hot_Index) - set(column))]

XcontinuousSet = XcontinuousPanda.to_numpy()
XbinarySet = XbinaryPanda.to_numpy()

#training/testing x values split by continuous and binary features
TrainingContinuousX = np.split(XcontinuousSet, [370])[0]
TestingContinuousX = np.split(XcontinuousSet, [370])[1]
TrainingBinaryX = np.split(XbinarySet, [370])[0]
TestingBinaryX = np.split(XbinarySet, [370])[1]




XSet = XPanda2.to_numpy() #unsplit x attributes, np form

YClass = YPanda.to_numpy()  

AllIndex = YPanda.value_counts().index.tolist()

YClass = np.where(np.isin(YClass, cm.pos_value), 1, YClass)
YClass = np.where(np.isin(YClass, list(set(AllIndex) - set(cm.pos_value))), 0, YClass)

TrainingX = np.split(XSet, [370])[0]
TestingX = np.split(XSet, [370])[1]

TrainingY = np.split(YClass, [370])[0]
TrainingY.shape = [370,1]
TestingY = np.split(YClass, [370])[1]
TestingY.shape = [320,1]








"""
training = np.split(cm.data.to_numpy(),[370])[0]
testing = np.split(cm.data.to_numpy(),[370])[1]

ClassIndex = training.shape[1]-1

AllIndex = cm.data.iloc[:, ClassIndex].value_counts().index.tolist()

training[:, ClassIndex] = np.where(np.isin(training[:, ClassIndex], cm.pos_value), 1, training[:, ClassIndex])
training[:, ClassIndex] = np.where(np.isin(training[:, ClassIndex], list(set(AllIndex) - set(cm.pos_value))), 0, training[:, ClassIndex])

testing[:, ClassIndex] = np.where(np.isin(testing[:, ClassIndex], cm.pos_value), 1, testing[:, ClassIndex])
testing[:, ClassIndex] = np.where(np.isin(testing[:, ClassIndex], list(set(AllIndex) - set(cm.pos_value))), 0, testing[:, ClassIndex])

"""

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
index = np.random.choice(370,100,replace=False)
lr = LogisticRegression()
x = np.float64(TrainingX[index])
y = np.int32(TrainingY[index])
