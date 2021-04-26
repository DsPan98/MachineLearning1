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
    
    def fit(self,x,y,learning_rate=0.001,eps=1e-2):
        m = len(y)
        self.w = np.float64(np.zeros((x.shape[1],1)))
        g = np.inf
        it = 0
        while np.linalg.norm(g) > eps:
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
lr = LogisticRegression()
x = TrainingX
y = np.int32(TrainingY)
