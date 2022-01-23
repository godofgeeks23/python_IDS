How to run:
    python IDS_test.py 'path for the test file'
Output:
    The output is result.csv file, which contains two columns - TIME and LABEL.

Procedure - 
    1. First we have combined the dataset 1 and 2 given in the original question, into one database, called as the mod_data.csv, and added a last column called 'target' to indicate the data as nominal/attack.
    2. Then we cleaned the CSV file, by seperating timestamp into time, day, date, month and year, removed 0 and NaN values.
    3. Then we used Univariate selection to select the best features.
    4. Now, with the best features, we haved trained the model by KMeans clustering.
    5. We have trained model by dividing the data into training and testing data by various ratios, and saving the best ratio to get the maximum accuracy.
    