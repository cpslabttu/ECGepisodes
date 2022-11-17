import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression
from sklearn.feature_selection import SelectKBest, SelectPercentile

data = pd.read_csv('C:/Users/as097/Desktop/tsfelFeatures.csv', nrows = 20000)
#print(data.head())
X = data.drop('Label', axis = 1)
y = data['Label']
#print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)

#Remove constant, quasi constant, and duplicate features
constant_filter = VarianceThreshold(threshold=0.01)
constant_filter.fit(X_train)
X_train_filter = constant_filter.transform(X_train)
X_test_filter = constant_filter.transform(X_test)
X_train_T = X_train_filter.T
X_test_T = X_test_filter.T
X_train_T = pd.DataFrame(X_train_T)
X_test_T = pd.DataFrame(X_test_T)
print(X_train_T.duplicated().sum())

duplicated_features = X_train_T.duplicated()
features_to_keep = [not index for index in duplicated_features]
X_train_unique = X_train_T[features_to_keep].T
X_test_unique = X_test_T[features_to_keep].T
print(X_train_unique.shape, X_test_unique.shape)

#Calculate the MI
mi = mutual_info_classif(X_train_unique, y_train)
print(len(mi))

print(mi)

mi = pd.Series(mi)
mi.index = X_train_unique.columns
mi.sort_values(ascending=False, inplace = True)
mi.plot.bar(figsize = (16,5))
#plt.show()

sel = SelectKBest(mutual_info_regression, k = 9).fit(X_train, y_train)
print(X_train.columns[sel.get_support()])

print(len(X_train_unique.columns[sel.get_support()]))
help(sel)
X_train_mi = sel.transform(X_train_unique)
X_test_mi = sel.transform(X_test_unique)
print(X_train_mi.shape)

