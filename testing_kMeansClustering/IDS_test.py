import sys

import pandas as pd


try:
    file = open(sys.argv[1], "rb")
    print("File opened")
    data = pd.read_csv(sys.argv[1])
    

except FileNotFoundError:
    print("specified file was not found")


# with open('model.pickle', "rb") as file:
#     saved_model = pickle.load(file)
# savedmodelpredictions = saved_model.predict(test_data_X)
# print('Accuracy: {}'.format(100*accuracy_score(test_data_Y, savedmodelpredictions)))
