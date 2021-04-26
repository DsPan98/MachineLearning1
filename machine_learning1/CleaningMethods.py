#some ideas implemented from https://github.com/
#WillKoehrsen/feature-selector/blob/master/feature_selector/feature_selector.py


import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import matplotlib.pyplot as plt


"""
for the module Cleaning methods, there are multiple methods implemented, in order to aid data cleaning
####################
####################
The following methods are used to help delete selected features:
#1. Missing data
Features with missing data higher than a given threshold (0.85 by default) will be deleted
#2. High similarity
Features that have similarities of all its values, higher than a given threshold(0.9 by default) will be deleted
#3. Collinear
If multiple feature have a high correlation, it would be detected and removed
#4. Ordinal
If a feature is a ordinal or sequential presentation of another feature, it will be removed(i.e. 1st, 2nd, mid, high...)
#5. Low, zero importance
The importance of a feature is calculated as follow:
 1. take the positive point percentage as mean
 2. calculate the standard deviation of every feature,
     μ = pos% of training set
     xi = ith unique value in the feature, and the pos% of selected set with this unique value
     n = number of unique features
 3. the higher the importance rate, the higher the value of feature
 4. if the importance rate is lower than a given threshold (very close to 0), then it means every
     unique value in this feature have similar pos/neg distribution percentage with the overall training
     set, therefore is considered less important
 5. zero importance indicate they contribute nothing to the final attribute, therefore should be discarded
####################
The following methods are used to contain the feature, but alter certain values
#6. Fill missing
Features that have few missing data, would be taken care of by
    either given a reasonable guess, the mean (for continuous)
    the value with selected pos% closest to training set pos% (for categorical)
    or discard the single point
#7. Abnormal point
    abnormal continuous data point will be fixed
####################
####################
"""


class CleaningMethods():
    def __init__ (self, data, det_column, pos_value = ['g','+','pos']):
        self.data = data
        self.column_attribute = list(data.columns)
        self.det_column = det_column        #feature determining p/n
        self.pos_value = pos_value          #value determining to be positive
        self.pos_df = self.data.loc[self.data[self.det_column].isin(self.pos_value)]
                                            #the dataframe of all positive values
        self.pos_rate = self.pos_df.shape[0] / self.data.shape[0]
        
        #--------------------------
        # for missing method
        self.missing_threshold = None
        self.missing_indication = None
        self.record_missing = []            #in the form [[missing ratio, column name],...]
        self.record_missing2 =[]            #in the form [column names]

        #--------------------------
        #for similar method
        self.record_similar = []            #in the form [[value, similarity num(0-1), column name],...]
        self.record_similar2 = []           #in the form [column names]
        self.similar_threshold = None


        #--------------------------
        #for correlation method
        self.correlation_threshold = None
        self.one_hot_correlated = False     #whether using one hot encoding
        self.data_all = None                #the data after onehotencoding
        self.record_collinear = []          #in the form: feature, corr feature, corr value
        self.to_drop = None                 #in the form ['columnName valueX','columnName valueY'....]
        self.one_hot_features = None        #in the form ['column1_value1', 'column2_value2',...'columnx_valuey']
        self.corr_matrix = None             #the correlation matrix for all data (numerical + onehotencoding)

        #--------------------------
        #for ordinal method
        self.categorical_column = None      
        self.categorial_data = None
        self.corr_matrix_forOrdinal = None
        self.to_drop_ordinal = None
        self.record_ordinal = []

        #--------------------------
        #for normalize method
        self.record_normalize = []          #record the feature being normalized

        #--------------------------
        #for low importance method
        self.record_low = []                #record the features having low s.d. than the given threshold
        self.record_low2 = []               #in the form: [[feature, s.d.],...]
        self.sd_threshold = None            #the given threshold, by default 0.01

        #--------------------------
        #for high importance method
        self.record_high = []               #record the features having high s.d. than the given threshold
        self.record_high2 = []              #in the form: [[feature, s.d.],...]
        self.sd_threshold_high = None       #the given threshold, by default 0.5

        #--------------------------
        #for statistics
        self.selected_feature = None
        self.feature = None

                
    def missing(self, missing_threshold = 0.85, missing_indication = None):
        """
        if the misssing data above the threshold, then the feature is shown
        missing indication is the representation note for missing data, in adult
        example indication would be ' ?'
        """
        #missing_threshold: float between 0 - 1
        #missing_indication: the string or character, None for default
        
        
        self.missing_threshold = missing_threshold
        self.missing_indication = missing_indication
        self.record_missing = []
        self.record_missing2 = []
        
        for x in range(self.data.shape[1]):
            a = self.data.iloc[:, x].value_counts(normalize = True)
            b = a.index.tolist()
            for y in range(a.count()):
                if b[y] == self.missing_indication:
                    c = a.tolist()
                    d = c[y]
                    if d > self.missing_threshold:
                        self.record_missing.append([d, self.data.columns[x]])
                        self.record_missing2.append(self.data.columns[x])

        #print (self.record_missing)
        #print(len( self.record_missing))
        print('%d features identified with greater than %0.2f percent of missing values in the given data.' %(len(self.record_missing), self.missing_threshold * 100))
        if(len(self.record_missing) > 0):
            print('These features are:')
            print(self.record_missing2)
            print('And with information in the form: [ [missing ratio, feature name],... ]')
            print(self.record_missing)


    def high_similarity(self, similar_threshold = 0.9):
        """
        If any value in any column have an occurrence higher than a given threshold
        it will be detected, and the column name will be recorded
        """
        #similar_threshold: float between 0 - 1
        #When similar_threshold == 1, works as a unique identifier
        
        

        self.similar_threshold = similar_threshold
        self.record_similar = []
        self.record_similar2 = []
        
        for x in range(self.data.shape[1]):
            a = self.data.iloc[:, x].value_counts(normalize = True, sort = True)
            b = a.index.tolist()
            c = a.tolist()
            for y in range(a.count()):
                d = c[y]
                if d >= self.similar_threshold:
                    self.record_similar.append([b[y],d,self.data.columns[x]])
                    self.record_similar2.append(self.data.columns[x])
                elif d < self.similar_threshold:
                    break


        if similar_threshold == 1:
            print('%d features identified with identical values in the given data.' %(len(self.record_similar)))
            if(len(self.record_similar)>0):
                print('These features are:')
                print(self.record_similar2)

        if similar_threshold < 1:  
            print('%d features identified with %0.2f percent of similar values in the given data' %(len(self.record_similar), self.similar_threshold * 100))
            if(len(self.record_similar)>0):
                print('These features are:')
                print(self.record_similar2)
                print('And with information in the form: [[value, similarity ratio(0-1), feature name],...]')         
                print(self.record_similar)


    def collinear(self, correlation_threshold = 0.9, one_hot = False):
        #adopted from
        #some ideas implemented from https://github.com/
        #WillKoehrsen/feature-selector/blob/master/feature_selector/feature_selector.py
        
        """
        checks the correlation of all columns and return column names with higher correlation
        """
        #correlation_threshold: float between 0 - 1
        #one_hot: boolean, default false

        
        self.correlation_threshold = correlation_threshold
        self.one_hot_correlated = one_hot
        self.record_collinear = []
        
        if one_hot:
            #if doing one_hot, devide the data into multiple new column spaces, and calculate all correlation
            features = pd.get_dummies(self.data)
            self.one_hot_features = [column for column in features.columns if column not in self.column_attribute]
            #self.data_all = pd.concat([features[self.one_hot_features], self.data], axis = 1)
            self.data_all = features
            corr_matrix = pd.get_dummies(features).corr()
        else: corr_matrix = self.data.corr()
        
        self.corr_matrix = corr_matrix
        #get the correlation matrix
        
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))
        #extract the upper triangle of the matrix, as the other half have the same data
        to_drop = [column for column in upper.columns if any(upper[column].abs() > correlation_threshold)]
        record_collinear = pd.DataFrame(columns = ['drop_attribute', 'corr_attribute', 'corr_value'])

        for column in to_drop:
            
            corr_features = list(upper.index [upper[column].abs() > correlation_threshold])
            corr_value = list(upper[column][upper[column].abs() > correlation_threshold])
            drop_features = [column for _ in range(len(corr_features))]

            temp_df = pd.DataFrame.from_dict({'drop_attribute': drop_features, 'corr_attribute': corr_features, 'corr_value': corr_value})
            record_collinear = record_collinear.append(temp_df, ignore_index = True)
            
        self.record_collinear = record_collinear
        self.to_drop = to_drop
        print('%d features with a correlation magnitude greater than %0.2f.'% (len(to_drop), self.correlation_threshold))
        if(len(to_drop)>0):
            print('With the indexes:')
            print(to_drop)



    def ordinal(self):
        """
        checking for features that are ordinal, for example, in the adult census case, 'education-num' is the index for education
        therefore one of the features can be removed
        """
        
        self.categorical_column = self.data.select_dtypes(exclude = ["number", "bool_"]).columns
        self.categorial_data = self.data[self.categorical_column]
        self.record_ordinal = []

        
        enc = OrdinalEncoder()
        X = self.categorial_data
        enc.fit(X)
        b = enc.transform(X)
        df = pd.DataFrame(data = b, columns = self.categorical_column)
        df2 = pd.concat([self.data._get_numeric_data(), df], axis = 1)
        self.corr_matrix_forOrdinal = df2.corr()
        corr_matrix = self.corr_matrix_forOrdinal.abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))
        to_drop_ordinal = [column for column in upper.columns if any(upper[column].abs() > 0.8)]
        ordinal = pd.DataFrame(columns = ['drop_attribute', 'corr_attribute', 'corr_value'])

        for column in to_drop_ordinal:
            
            corr_features = list(upper.index [upper[column].abs() > 0.8])
            corr_value = list(upper[column][upper[column].abs() > 0.8])
            drop_features = [column for _ in range(len(corr_features))]

            temp_df = pd.DataFrame.from_dict({'drop_attribute': drop_features, 'corr_attribute': corr_features, 'corr_value': corr_value})
            ordinal = ordinal.append(temp_df, ignore_index = True)
            
        self.record_ordinal = ordinal
        self.to_drop_ordinal = to_drop_ordinal
        print('%d features show ordinal features.'% (len(to_drop_ordinal)))
        if(len(to_drop_ordinal)>0):
            print(to_drop_ordinal)


    def normalize(self, normalize_threshold = 1000):
        """
        This method is implemented when encountering features with mass value
        The data would then be reduced to a normalized value to [0, 1]
        """
        self.record_normalize = []
        a = self.data._get_numeric_data()
        for x in range(a.shape[1]):
            b = a.iloc[:, x]
            if b.max() > normalize_threshold:
                c = b.name
                self.data[[c]] = StandardScaler().fit_transform(self.data[[c]])
                self.record_normalize.append(c)

        print('%d features above the threshold: %d, have been successfully normalized.' %(len(self.record_normalize), normalize_threshold))
        if(len(self.record_normalize) > 0):
            print(self.record_normalize)
            print('#######################################')
            print('Successfully Normalized Features!')
            print('#######################################')

    def low_importance(self, sd_threshold = 0.01):
        
    
        """
        Check all existing data, for every feature, calculate the
        standard deviation using pos/neg rate as mean
        (e.g. 900 sample, 450 pos, 450 neg, mean = 0.5)
        and calculate the sum average of every feature, delete those with
        a lower sd than the given threshold
        """
        self.sd_threshold = sd_threshold
        self.record_low = []
        self.record_low2 = []
        
        for x in range(len(self.column_attribute)):
            pos_data = self.pos_df.iloc[:, x].value_counts()
            all_data = self.data.iloc[:, x].value_counts()
            pos_index = pos_data.index.tolist()
            all_index = all_data.index.tolist()

            sd = 0
            for y in range (len(pos_index)):
                pos_per = pos_data[[pos_index[y]]] / all_data[[pos_index[y]]]
                sd += (pos_per - self.pos_rate).tolist()[0] ** 2

            sd = (sd / (len(pos_index))) **(0.5)
            if sd < sd_threshold:
                self.record_low.append(self.column_attribute[x])
                self.record_low2.append([self.column_attribute[x], sd])
           
             # the feature would have very little influence
             # therefore could be deleted
        print('%d features below the threshold: %0.2f, indicating they have low importance' %(len(self.record_low), self.sd_threshold))
        if (len(self.record_low) > 0):
            print('These features are:')
            print(self.record_low)
            print('And with information in the form: [[feature name, standard deviation],...]')
            print(self.record_low2)
            
         
    def zero_importance(self):
        """
        Implemented simply using low importance method
        """
        self.record_low = []
        self.record_low2 = []
        self.low_importance(sd_threshold = 0)


    
    def high_importance(self, high_threshold = 0.5):
        """
        for certain features that have a higher importance
        it would be singled out, maybe weigh more in further process
        For such features, even if they experience missing data, 
        """

        self.record_high = []
        self.record_high2 = []
        self.sd_threshold_high = high_threshold
                
        for x in range(len(self.column_attribute)):
            pos_data = self.pos_df.iloc[:, x].value_counts()
            all_data = self.data.iloc[:, x].value_counts()
            pos_index = pos_data.index.tolist()
            all_index = all_data.index.tolist()

            sd = 0
            for y in range (len(pos_index)):
                pos_per = pos_data[[pos_index[y]]] / all_data[[pos_index[y]]]
                sd += (pos_per - self.pos_rate).tolist()[0] ** 2

            sd = (sd / (len(pos_index))) **(0.5)
            if sd > self.sd_threshold_high:
                self.record_high.append(self.column_attribute[x])
                self.record_high2.append([self.column_attribute[x], sd])
           
             #These features have very little influence, therefore could be deleted
        print('%d features higher than the threshold: %0.2f, indicating they have more weight in the matter' %(len(self.record_high), self.sd_threshold_high))
        if (len(self.record_high) > 0):
            print('These features are:')
            print(self.record_high)
            print('And with information in the form: [[feature name, standard deviation],...]')
            print(self.record_high2)
            print()
            
        
    #def duplicate(self)：
     #   self.data.drop_duplicates(subset=self.column_attribute)


    def delete(self, all = False, use_zero_importance = False, use_low_importance = False, use_ordinal = False, use_missing = False, use_high_similarity = False, use_collinear = False, correlation_threshold = 0.8, missing_threshold = 0.85, missing_indication = None, similar_threshold = 0.9, one_hot = False, sd_threshold = 0.01):
        if all == True:
            use_missing == True
            use_high_similarity == True
            use_collinear == True
            use_ordinal == True
            use_low_importance == True

        self.high_importance()
        a = self.record_high
        
       
           
        if use_missing == True:
            self.missing(missing_threshold, missing_indication)
            self.data.drop(list(set(self.record_missing2) - set(a)), inplace = True, axis = 1)
            self.__init__(data = self.data, det_column = self.det_column, pos_value = self.pos_value)
            
        if use_high_similarity == True:
            self.high_similarity(similar_threshold)
            self.data.drop(list(set(self.record_similar2) - set(a)), inplace = True, axis = 1)
            self.__init__ (data = self.data, det_column = self.det_column, pos_value = self.pos_value)

        if use_collinear == True:
            self.collinear(correlation_threshold, one_hot)
            self.__init__ (data = self.data, det_column = self.det_column, pos_value = self.pos_value)
            
        if use_ordinal == True:
            self.ordinal()
            self.data.drop(list(set(self.to_drop_ordinal) - set(a)), inplace = True, axis = 1)
            self.__init__ (data = self.data, det_column = self.det_column, pos_value = self.pos_value)
            
        if use_low_importance == True:
            self.low_importance(sd_threshold)
            self.data.drop(list(set(self.record_low) - set(a)), inplace = True, axis = 1)
            self.__init__ (data = self.data, det_column = self.det_column, pos_value = self.pos_value)

        if use_zero_importance == True & use_low_importance == False:
            self.zero_importance()
            self.data.drop(list(set(self.record_low) - set(a)), inplace = True, axis = 1)
            self.__init__ (data = self.data, det_column = self.det_column, pos_value = self.pos_value)

        print('#######################################')
        print('Successfully Deleted Unwanted Features!')
        print('#######################################')
        

    


    def fill_missing(self, missing_indication = None):
        self.missing_indication = missing_indication
        numeric_column = list(self.data._get_numeric_data().columns)
        categorical_column = list(set(self.data.columns) - set(self.data._get_numeric_data().columns))
        num = 0
        for x in range(len(numeric_column)): #use mean
            a = self.data[numeric_column[x]]
            mean = a.mean(axis = 0)
            for y in range(a.count()):
                if a[y] == self.missing_indication:
                    
                    a[y] = mean
                    num += 1

        for k in range(len(categorical_column)): #find categorical value closest to pos%
            
            a = self.data[categorical_column[k]]

            pos_data = self.pos_df[categorical_column[k]].value_counts()
            all_data = a.value_counts()
            pos_index = pos_data.index.tolist()
            all_index = all_data.index.tolist()

            sd = []
            if self.missing_indication in all_data:
                for m in range (len(pos_index)):
                    pos_per = pos_data[[pos_index[m]]] / all_data[[pos_index[m]]]
                    sd.append(list(pos_per)[0])
        

                fill_categorical = pos_index[min(range(len(sd)), key = lambda i: abs(sd[i] - self.pos_rate))]
                #should get the index of value closest to pos%

            
                num += all_data[self.missing_indication]
                self.data[categorical_column[k]] = self.data[categorical_column[k]].replace(self.missing_indication, fill_categorical)
                #if self.missing_indication != fill_categorical:
        print('#######################################')
        print('Successfully Filled All Missing Values')
        print('#######################################')


    def statistics(self, selected_feature):
        self.selected_feature = selected_feature
        print('The distribution for positive and negative class is:')
        print('Positive Class = %0.3f | Negative Class = %0.3f'%(self.pos_rate, (1 - self.pos_rate)))
        #pos/neg class distribution
        if(len(selected_feature) > 0):
            for x in range(len(selected_feature)):
                print('#######################################')
                print('The Selected Feature #%s# has distribution:' %(selected_feature[x]))
                for y in range(len(self.data[selected_feature[x]].value_counts(normalize = True, sort = True).tolist())):
                    print('Value "%s" has distribution %0.3f' %(self.data[selected_feature[x]].value_counts(normalize = True, sort = True).index.tolist()[y], self.data[selected_feature[x]].value_counts(normalize = True, sort = True).tolist()[y]))
                print('#######################################')
        #selected feature distribution
        if(len(selected_feature) > 0):
            feature = pd.get_dummies(self.data[selected_feature])
            print(feature.corr())
            self.feature = feature
        #plot data for selected feature
        g = sns.pairplot(self.feature)
      
        print(g)
        plt.show()
