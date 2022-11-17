import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold

data = pd.read_csv('C:/Users/as097/Desktop/PVC/tsfelWRM.csv', nrows = 20000)
#print(data.head())

#data = data[['0_Spectral roll-off','0_Standard deviation', '0_LPCC_4', '0_Total energy', '0_Spectral spread', '0_LPCC_11',  'Label']].copy()
#X = data.copy()
X = data.drop('Label', axis = 1)
y = data['Label']
#print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)

# Constant features removal
constant_filter = VarianceThreshold(threshold=0)
print(constant_filter.fit(X_train))
print(constant_filter.get_support().sum())
constant_list = [not temp for temp in constant_filter.get_support()]
print(constant_list)
print(X.columns[constant_list])

X_train_filter = constant_filter.transform(X_train)
X_test_filter = constant_filter.transform(X_test)
print(X_train_filter.shape, X_test_filter.shape, X_train.shape)

#Quasi constant feature removal
quasi_constant_filter = VarianceThreshold(threshold=0.01)
print(quasi_constant_filter.fit(X_train_filter))
print(quasi_constant_filter.get_support().sum())
#print(291-245)

X_train_quasi_filter = quasi_constant_filter.transform(X_train_filter)
X_test_quasi_filter = quasi_constant_filter.transform(X_test_filter)
print(X_train_quasi_filter.shape, X_test_quasi_filter.shape)
#print(370-245)

#Remove Duplicate Features
X_train_T = X_train_quasi_filter.T
X_test_T = X_test_quasi_filter.T
print(type(X_train_T))

X_train_T = pd.DataFrame(X_train_T)
X_test_T = pd.DataFrame(X_test_T)
print(X_train_T.shape, X_test_T.shape)
print(X_train_T.duplicated().sum())

duplicated_features = X_train_T.duplicated()
print(duplicated_features)

features_to_keep = [not index for index in duplicated_features]
X_train_unique = X_train_T[features_to_keep].T
X_test_unique = X_test_T[features_to_keep].T
print(X_train_unique.shape, X_train.shape)
#print(370-227)

#Build ML model and compare the performance of the selected feature
def run_randomForest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Accuracy on test set: ')
    print(accuracy_score(y_test, y_pred))
#time
(run_randomForest(X_train_unique, X_test_unique, y_train, y_test))
#time
(run_randomForest(X_train, X_test, y_train, y_test))
#print((1.51-1.26)*100/1.51)

corrmat = X_train_unique.corr()
plt.figure(figsize=(12,8))
sns.heatmap(corrmat)
#plt.show()

def get_correlation(data, threshold):
    corr_col = set()
    corrmat = data.corr()
    for i in range(len(corrmat.columns)):
        for j in range(i):
            if abs(corrmat.iloc[i, j])> threshold:
                colname = corrmat.columns[i]
                corr_col.add(colname)
    return corr_col
corr_features = get_correlation(X_train_unique, 0.85)
print(corr_features)
print(len(corr_features))
'''
corr_features = pd.Series(corr_features, index = corrmat.columns)
corr_features.sort_values(ascending = True, inplace = True)
corr_features.plot.bar(figsize = (16, 5))
plt.show()
'''
X_train_uncorr = X_train_unique.drop(labels=corr_features, axis = 1)
X_test_uncorr = X_test_unique.drop(labels = corr_features, axis = 1)
print(X_train_uncorr.shape, X_test_uncorr.shape)
#time
run_randomForest(X_train_uncorr, X_test_uncorr, y_train, y_test)
#time
run_randomForest(X_train, X_test, y_train, y_test)
#print((1.53-0.912)*100/1.53)
#print(corrmat)

corrdata = corrmat.abs().stack()
#print(corrdata)

corrdata = corrdata.sort_values(ascending=False)
#print(corrdata)

corrdata = corrdata[corrdata>0.85]
corrdata = corrdata[corrdata<1]
#print(corrdata)

corrdata = pd.DataFrame(corrdata).reset_index()
corrdata.columns = ['features1', 'features2', 'corr_value']
print(corrdata)

grouped_feature_list = []
correlated_groups_list = []
for feature in corrdata.features1.unique():
    if feature not in grouped_feature_list:
        correlated_block = corrdata[corrdata.features1 == feature]
        grouped_feature_list = grouped_feature_list + list(correlated_block.features2.unique()) + [feature]
        correlated_groups_list.append(correlated_block)
#print(len(correlated_groups_list))
#print(X_train.shape, X_train_uncorr.shape)
for group in correlated_groups_list:
    print(group)

#Feature Importance based on tree based classifiers
important_features = []
for group in correlated_groups_list:
    features = list(group.features1.unique()) + list(group.features2.unique())
    rf = RandomForestClassifier(n_estimators=100, random_state=0)
    rf.fit(X_train_unique[features], y_train)

    importance = pd.concat([pd.Series(features), pd.Series(rf.feature_importances_)], axis=1)
    importance.columns = ['features', 'importance']
    importance.sort_values(by='importance', ascending=False, inplace=True)
    feat = importance.iloc[0]
    important_features.append(feat)
print(important_features)

important_features = pd.DataFrame(important_features)
important_features.reset_index(inplace=True, drop = True)
print(important_features)

features_to_consider = set(important_features['features'])
features_to_discard = set(corr_features) - set(features_to_consider)
features_to_discard = list(features_to_discard)
X_train_grouped_uncorr = X_train_unique.drop(labels = features_to_discard, axis = 1)
#print(X_train_grouped_uncorr.shape)

X_test_grouped_uncorr = X_test_unique.drop(labels=features_to_discard, axis = 1)
#print(X_test_grouped_uncorr.shape)

#time
(run_randomForest(X_train_grouped_uncorr, X_test_grouped_uncorr, y_train, y_test))
#time
(run_randomForest(X_train, X_test, y_train, y_test))
#time
(run_randomForest(X_train_uncorr, X_test_uncorr, y_train, y_test))

# import SVC classifier
from sklearn.svm import SVC
# import metrics to compute accuracy
from sklearn.metrics import accuracy_score
# instantiate classifier with default hyperparameters
svc=SVC()
# fit classifier to training set
svc.fit(X_train,y_train)
# make predictions on test set
y_pred=svc.predict(X_test)
# compute and print accuracy score
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

# instantiate classifier with rbf kernel and C=100
svc=SVC(C=100.0)
# fit classifier to training set
svc.fit(X_train,y_train)
# make predictions on test set
y_pred=svc.predict(X_test)
# compute and print accuracy score
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

# instantiate classifier with rbf kernel and C=1000
svc=SVC(C=1000.0)
# fit classifier to training set
svc.fit(X_train,y_train)
# make predictions on test set
y_pred=svc.predict(X_test)
# compute and print accuracy score
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

# instantiate classifier with linear kernel and C=1.0
linear_svc=SVC(kernel='linear', C=1.0)
# fit classifier to training set
linear_svc.fit(X_train,y_train)
# make predictions on test set
y_pred_test=linear_svc.predict(X_test)
# compute and print accuracy score
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))

# instantiate classifier with linear kernel and C=100.0
linear_svc100=SVC(kernel='linear', C=100.0)
# fit classifier to training set
linear_svc100.fit(X_train, y_train)
# make predictions on test set
y_pred=linear_svc100.predict(X_test)
# compute and print accuracy score
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

# instantiate classifier with linear kernel and C=1000.0
linear_svc1000=SVC(kernel='linear', C=1000.0)
# fit classifier to training set
linear_svc1000.fit(X_train, y_train)
# make predictions on test set
y_pred=linear_svc1000.predict(X_test)
# compute and print accuracy score
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

y_pred_train = linear_svc.predict(X_train)
#print(y_pred_train)
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))

# print the scores on training and test set
print('Training set score: {:.4f}'.format(linear_svc.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(linear_svc.score(X_test, y_test)))

# instantiate classifier with polynomial kernel and C=1.0
poly_svc=SVC(kernel='poly', C=1.0)
# fit classifier to training set
poly_svc.fit(X_train,y_train)
# make predictions on test set
y_pred=poly_svc.predict(X_test)
# compute and print accuracy score
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

# instantiate classifier with polynomial kernel and C=100.0
poly_svc100=SVC(kernel='poly', C=100.0)
# fit classifier to training set
poly_svc100.fit(X_train, y_train)
# make predictions on test set
y_pred=poly_svc100.predict(X_test)
# compute and print accuracy score
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

# instantiate classifier with sigmoid kernel and C=1.0
sigmoid_svc=SVC(kernel='sigmoid', C=1.0)
# fit classifier to training set
sigmoid_svc.fit(X_train,y_train)
# make predictions on test set
y_pred=sigmoid_svc.predict(X_test)
# compute and print accuracy score
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

# instantiate classifier with sigmoid kernel and C=100.0
sigmoid_svc100=SVC(kernel='sigmoid', C=100.0)
# fit classifier to training set
sigmoid_svc100.fit(X_train,y_train)
# make predictions on test set
y_pred=sigmoid_svc100.predict(X_test)
# compute and print accuracy score
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))