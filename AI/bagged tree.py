#%matplotlib inline

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix , classification_report
from sklearn.model_selection import train_test_split

# Bagged Trees Regressor
from sklearn.ensemble import BaggingRegressor
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("C:/Users/as097/Desktop/PVC/tsfelWRM_multi.csv")
#print(df.sample(5))
#print(df.Label.value_counts())


df.drop('0_ECDF Percentile Count_0',axis='columns',inplace=True)
#print(df.dtypes)

X = df.drop('Label1',axis='columns')
y = testLabels = df.Label1.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=15, stratify=y)

'''
###### Bagged Tree Classifier with criterion gini index
def BT(X_train, y_train, X_test, y_test):
    reg = BaggingRegressor(n_estimators=100).fit(X_train, y_train)
    # Returns a NumPy Array
    # Predict for One Observation
    #reg.predict(X_test.iloc[0].values.reshape(1, -1))
    y_preds = reg.predict(X_test)
    #print(reg.predict(X_test[0:10]))
    score = reg.score(X_test, y_test)
    print(score)
    print("Classification Report: \n", classification_report(y_test, y_preds))
    return y_preds

y_preds = BT(X_train, y_train, X_test, y_test)

cm = confusion_matrix(y_test, y_preds)
print(cm)
'''
'''
reg = BaggingRegressor(n_estimators=100).fit(X_train, y_train)
# Returns a NumPy Array
# Predict for One Observation
#y_preds =reg.predict(X_test.iloc[0].values.reshape(1, -1))
y_preds = reg.predict(X_test)
#print(reg.predict(X_test[0:10]))
score = reg.score(X_test, y_test)
print(score)
print("Classification Report: \n", classification_report(y_test, y_preds))


# List of values to try for n_estimators:
estimator_range = [1] + list(range(10, 150, 20))

scores = []

for estimator in estimator_range:
    reg = BaggingRegressor(n_estimators=estimator, random_state=0)
    reg.fit(X_train, y_train)
    scores.append(reg.score(X_test, y_test))
plt.figure(figsize = (10,7))
plt.plot(estimator_range, y_preds);
plt.show()
plt.xlabel('n_estimators', fontsize =20);
plt.ylabel('Score', fontsize = 20);
plt.tick_params(labelsize = 18)
plt.grid()
'''

# explore bagging ensemble number of trees effect on performance
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import BaggingClassifier
from matplotlib import pyplot


# get a list of models to evaluate
def get_models():
    models = dict()
    # define number of trees to consider
    n_trees = [10, 50, 100, 500, 500, 1000, 5000]
    for n in n_trees:
        models[str(n)] = BaggingClassifier(n_estimators=n)
    return models


# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
    # define the evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate the model and collect the results
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores


# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
    # evaluate the model
    scores = evaluate_model(model, X, y)
    # store the results
    results.append(scores)
    names.append(name)
    # summarize the performance along the way
    print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()