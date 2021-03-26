import pickle
import os
import pandas as pd
import graphviz
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import  train_test_split
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import zipfile

file_url = "diagnosis.csv"

# use pandas to read data from csv
dataset = pd.read_csv(file_url)
# convert the pandas data frame into an array of values
array = dataset.values
# get the count of the columnes in the dataset
columns_count = len(dataset.columns.values)

# get all rows of every column except the last two, as these are the decision columns
X = array[:,0:columns_count-2]
# get all rows of just the lat two decision columns
Y = array[:,columns_count-2:columns_count]

labelencoder = preprocessing.LabelEncoder()

# convert our yes/no columns into integer values
# using the label encoder defined above
for column in range(X.shape[1]):
    X[:,column] = labelencoder.fit_transform(X[:,column])

# get the names of the diagnosis (decision variables)
first_label = dataset.columns.values[columns_count-2]
second_label = dataset.columns.values[columns_count-1]
# create array for storing new multi label classification tuples
Y_multilabel = []
for y in range(Y.shape[0]):
    labels = []

    # for the first diagnosis,
    # if yes then we append the name of the disease as a classification in the tuple
    if Y[y,0] == "yes":
        labels.append(first_label)
    # same as above, but for the second disease classification
    if Y[y,1] == "yes": 
        labels.append(second_label)
    Y_multilabel.append(labels)

mlb = MultiLabelBinarizer()
# convert all the labels into a multilabel binary form
Y = mlb.fit_transform(Y_multilabel)

# split our train and test data up 80% training and 20% testing
validation_size = 0.20
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = validation_size)

# declare both decision tree and random forest models, so that we can compare for the best
# model later on
RF = RandomForestClassifier(criterion="entropy", n_estimators=5, max_features=6, max_depth=6)
DT = DecisionTreeClassifier(criterion="gini", max_features=6, max_depth=5, min_samples_leaf=1, min_samples_split=4)

# fit our training observances to our training target variables
RF.fit(X_train, Y_train)
DT.fit(X_train, Y_train)

# predict our test target instances
# using our test observations
y_predict_DT = RF.predict(X_test)
y_predict_RF = DT.predict(X_test)


# get our accuracy score for our models
# by comparing test target values vs predicted values
RF_accuracy_score = accuracy_score(Y_test, y_predict_RF)
DT_accuracy_score = accuracy_score(Y_test, y_predict_DT)

print("Random Forest Accuracy:{0}".format(RF_accuracy_score))
print("Decision Tree Accuracy:{0}".format(DT_accuracy_score))

# print out our confusion matrices
print(multilabel_confusion_matrix(Y_test, y_predict_DT))
print(multilabel_confusion_matrix(Y_test, y_predict_RF))

with open("random_forest_diagnosis_model.pkl", 'wb') as f:
    pickle.dump(RF, f)

zipfile.ZipFile("model.zip", mode="w").write("random_forest_diagnosis_model.pkl")