with open('model.pickle', "rb") as file:
    saved_model = pickle.load(file)
savedmodelpredictions = saved_model.predict(test_data_X)
print('Accuracy: {}'.format(100*accuracy_score(test_data_Y, savedmodelpredictions)))
