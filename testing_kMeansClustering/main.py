import sys
from cgi import test
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


data = pd.read_csv("mod_dataset/mod_data.csv")

data[['DATE','TIME']] = data['DATETIME'].str.split(' ',expand=True)
data['DATETIME'] = data['DATE']
data['DATETIME'] = pd.to_datetime(data['DATETIME'], format='%d/%m/%y')
data['DATE_DAY'] = data['DATETIME'].dt.day
data['DATE_MONTH'] = data['DATETIME'].dt.month
data['DATE_YEAR'] = data['DATETIME'].dt.year
cols = data.columns.tolist()
cols = cols[-5:] + cols[:-5]
data = data[cols]
data = data.drop('DATE', 1)
data = data.drop('DATETIME', 1)
# data.to_csv("mod_dataset/mod_data2.csv", index=False)

data = data.apply(pd.to_numeric)
data[data < 0] = 0

X = data.iloc[:,0:47]  #independent columns
y = data.iloc[:,-1]    #target column


vari = 10
bestfeatures = SelectKBest(score_func=chi2, k=vari)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']
# print(featureScores.nlargest(10,'Score'))
# print(featureScores)


new_data_X = pd.DataFrame(data[list(featureScores.nlargest(vari,'Score')['Specs'])])
new_data_Y = pd.DataFrame(data['target'])
new_data = pd.concat([new_data_X,new_data_Y],axis=1)

train_data = new_data.iloc[0:9057,:]
train_data_X = train_data.iloc[:,0:47]
train_data_Y = train_data.iloc[:,-1]

test_data = new_data.iloc[9057:,:]
test_data_X = test_data.iloc[:,0:47]
test_data_Y = test_data.iloc[:,-1]


# kmeans = KMeans(2)
# kmeans.fit(train_data)
# result = kmeans.predict(new_data_X)


kmeans = KMeans(2)
kmeans.fit(train_data_X, train_data_Y)
predictions = kmeans.predict(test_data_X)

print(classification_report(test_data_Y, predictions))
print('Accuracy: {}'.format(accuracy_score(test_data_Y, predictions)))

