import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

data = pd.read_csv("mod_dataset/mod_data.csv")

# data['DATETIME'] = pd.to_datetime(data['DATETIME'])
# print(data['DATETIME'].head())

data[['DATE','TIME']] = data['DATETIME'].str.split(' ',expand=True)
data['DATETIME'] = data['DATE']
data['DATETIME'] = pd.to_datetime(data['DATETIME'], format='%d/%m/%y')
data['DATE_DAY'] = data['DATETIME'].dt.day
data['DATE_MONTH'] = data['DATETIME'].dt.month
data['DATE_YEAR'] = data['DATETIME'].dt.year
cols = data.columns.tolist()
cols = cols[-5:] + cols[:-5]
data = data[cols]
# print(cols)
data = data.drop('DATE', 1)
data = data.drop('DATETIME', 1)
# data.to_csv("mod_dataset/mod_data2.csv", index=False)

data = data.apply(pd.to_numeric)
data[data < 0] = 0
X = data.iloc[:,0:47]  #independent columns
y = data.iloc[:,-1]    #target column, apply SelectKBest class to extract top 10 best features

bestfeatures = SelectKBest(score_func=chi2, k=20)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(20,'Score'))  #print 10 best features
