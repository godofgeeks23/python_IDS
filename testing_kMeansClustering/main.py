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
import pickle


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

# best feature selection
vari = 14
bestfeatures = SelectKBest(score_func=chi2, k=vari)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Feature','Score']


new_data_X = pd.DataFrame(data[list(featureScores.nlargest(vari,'Score')['Feature'])])
new_data_Y = pd.DataFrame(data['target'])
new_data = pd.concat([new_data_X,new_data_Y],axis=1)

traintesttratio = 0.02
max_accuracy = 0
while(traintesttratio<1):
    dissection = int(len(new_data) * traintesttratio)
    train_data = new_data.iloc[0:dissection,:]
    train_data_X = train_data.iloc[:, train_data.columns != 'target']
    train_data_Y = train_data.iloc[:,-1]

    test_data = new_data.iloc[dissection:,:]
    test_data_X = test_data.iloc[:, test_data.columns != 'target']
    test_data_Y = test_data.iloc[:,-1]


    kmeans = KMeans(2)
    kmeans.fit(train_data_X, train_data_Y)
    predictions = kmeans.predict(test_data_X)

    # # print(classification_report(test_data_Y, predictions))
    # print('Accuracy: {}'.format(accuracy_score(test_data_Y, predictions))*100)
    traintesttratio += 0.01
    if(accuracy_score(test_data_Y, predictions) > max_accuracy):
        max_accuracy = accuracy_score(test_data_Y, predictions)
print('Max Accuracy: {}'.format(max_accuracy*100))

# print ("Model trained. Saving model to model.pickle")
# with open("model.pickle", "wb") as file:
#     pickle.dump(kmeans, file)
# print ("Model saved.")

# with open('model.pickle', "rb") as file:
#     saved_model = pickle.load(file)
# savedmodelpredictions = saved_model.predict(test_data_X)
# print('Accuracy: {}'.format(100*accuracy_score(test_data_Y, savedmodelpredictions)))