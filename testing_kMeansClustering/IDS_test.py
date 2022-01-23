# import the libraries used
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
import sys

# load the data for training and testing the model
data = pd.read_csv("mod_dataset/mod_data.csv")

# clean the data
data[['DATE', 'TIME']] = data['DATETIME'].str.split(' ', expand=True)
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
data = data.drop('TIME', 1)
data = data.drop('DATE_DAY', 1)
data = data.drop('DATE_MONTH', 1)
data = data.drop('DATE_YEAR', 1)
# data.to_csv("mod_dataset/mod_data2.csv", index=False)

# fix all the negative values to 0 and any NaN values to numeric
data = data.apply(pd.to_numeric)
data[data < 0] = 0

X = data.iloc[:, data.columns != 'target']  # independent columns
y = data.iloc[:, -1]  # target column

# best feature selection - by univariate selection
vari = 10   # number of features to select
bestfeatures = SelectKBest(score_func=chi2, k=vari)
fit = bestfeatures.fit(X, y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Feature', 'Score']

# # reducing the whole dataset to the selected features
# new_data_X = pd.DataFrame(
#     data[list(featureScores.nlargest(vari, 'Score')['Feature'])])
# new_data_Y = pd.DataFrame(data['target'])
# new_data = pd.concat([new_data_X, new_data_Y], axis=1)
# # print(new_data.head())

# # training the model
# traintesttratio = 0.02
# max_accuracy = 0
# while(traintesttratio < 1):
#     dissection = int(len(new_data) * traintesttratio) # number of rows to be used for training
#     train_data = new_data.iloc[0:dissection, :]   # training data
#     train_data_X = train_data.iloc[:, train_data.columns != 'target'] # independent columns
#     train_data_Y = train_data.iloc[:, -1]   # target column

#     test_data = new_data.iloc[dissection:, :] # testing data
#     test_data_X = test_data.iloc[:, test_data.columns != 'target']    # independent columns
#     test_data_Y = test_data.iloc[:, -1]   # target column

#     kmeans = KMeans(2)    # create an instance of the model (2 clusters)
#     kmeans.fit(train_data_X, train_data_Y)    # train the model
#     predictions = kmeans.predict(test_data_X) # test the model

#     # # print(classification_report(test_data_Y, predictions))
#     traintesttratio += 0.001  # increase the traintesttratio
#     if(accuracy_score(test_data_Y, predictions) > max_accuracy):  # if the accuracy is better than the previous one
#         max_accuracy = accuracy_score(test_data_Y, predictions)   # save the new accuracy
#         best_kmeans = kmeans  # save the model
# print('Accuracy achieved: {}'.format(max_accuracy*100))   # print the accuracy
# picklefilename = "best_model_nontime{}.pickle".format(max_accuracy*100)   # save the best model
# with open(picklefilename, "wb") as file:
#     pickle.dump(best_kmeans, file)    

# the model has been trained from the above code. now we need to use it to predict for the file that was passed as an argument
try:    # try to open the file
    resultinput = open(sys.argv[1], "rb")   # open the file
    print("File opened")
    resultinput.close()

    resultinputfile = pd.read_csv(sys.argv[1])  # read the file
    resultinputdata = pd.DataFrame(resultinputfile[list(featureScores.nlargest(vari, 'Score')['Feature'])]) # select the best features (data refining)
    with open('best_model_nontime92.3076923076923.pickle', "rb") as file:   # load the best saved model
        saved_model = pickle.load(file)
    finalresult = pd.DataFrame(resultinputfile['INDEX(TIME_IN_HOURS)']) # create a dataframe for the result
    finalresult['TIME'] = finalresult['INDEX(TIME_IN_HOURS)']
    finalresult.drop('INDEX(TIME_IN_HOURS)', 1, inplace=True)
    finalresult['LABEL'] = saved_model.predict(resultinputdata)    # predict the result
    finalresult['LABEL'].replace(to_replace=0, value='NORMAL', inplace=True)    # replace the 0 values with NORMAL
    finalresult['LABEL'].replace(to_replace=1, value='ATTACK', inplace=True)    # replace the 1 values with ATTACK
    finalresult.to_csv("result.csv", index=False)  # save the result

except FileNotFoundError:   # if the file is not found
    print("specified file was not found")
