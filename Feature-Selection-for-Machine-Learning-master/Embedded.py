import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
#%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import f_classif, f_regression


data = pd.read_csv('C:/Users/as097/Desktop/PVC/tsfelWRM.csv')
#print(data.head())

#print(data.isnull().sum())


data.drop(labels = ['0_FFT mean coefficient_0','0_FFT mean coefficient_1'], axis = 1, inplace = True)
data = data.dropna()
#print(data.isnull().sum())

#data = data[['0_Spectral roll-off','0_Standard deviation', 'Label']].copy()
#data = data[['0_Spectral roll-off','0_Standard deviation','0_LPCC_4', 'Label']].copy()
#data = data[['0_Spectral roll-off','0_Standard deviation','0_LPCC_4', '0_Total energy', 'Label']].copy()
#data = data[['0_Spectral roll-off','0_Standard deviation','0_LPCC_4','0_Total energy','0_Spectral spread', 'Label']].copy()
#data = data[['0_Spectral roll-off','0_Standard deviation','0_LPCC_4','0_Total energy','0_Spectral spread','0_LPCC_11', 'Label']].copy()
#data = data[['0_Spectral roll-off','0_Standard deviation','0_LPCC_4','0_Total energy','0_Spectral spread','0_LPCC_11', '0_FFT mean coefficient_8', 'Label']].copy()
#data = data[['0_Spectral roll-off','0_Standard deviation','0_LPCC_4','0_Total energy','0_Spectral spread','0_LPCC_11','0_FFT mean coefficient_8', '0_FFT mean coefficient_5','Label']].copy()
#data = data[['0_Spectral roll-off','0_Standard deviation','0_LPCC_4','0_Total energy','0_Spectral spread','0_LPCC_11','0_FFT mean coefficient_8', '0_FFT mean coefficient_5','0_LPCC_2','Label']].copy()
#data = data[['0_Spectral roll-off','0_Standard deviation','0_LPCC_4','0_Total energy','0_Spectral spread','0_LPCC_11','0_FFT mean coefficient_8', '0_FFT mean coefficient_5','0_LPCC_2','0_LPCC_3','Label']].copy()
#print(data.head())
#print(data.isnull().sum())

#X = data.copy()
X = data.drop('Label', axis = 1)
y = data['Label']
#print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 43)
sel = SelectFromModel(LinearRegression())
print(sel.fit(X_train, y_train))
print(sel.get_support())
print(sel.estimator_.coef_)

mean = np.mean(np.abs(sel.estimator_.coef_))
#print(mean)

print(np.abs(sel.estimator_.coef_))

features = X_train.columns[sel.get_support()]
print(features)

X_train_reg = sel.transform(X_train)
X_test_reg = sel.transform(X_test)
#print(X_test_reg.shape)

# feature ranking
se = f_classif(X_train, y_train)
p_values = pd.Series(se[1])
p_values.index = X_train.columns
p_values.sort_values(ascending = True, inplace = True)
p_values.plot.bar(figsize = (16, 5))
#plt.show()

def run_randomForest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Accuracy: ', accuracy_score(y_test, y_pred))
    print(metrics.classification_report(y_pred, y_test))
    rf_matrix = confusion_matrix(y_test, y_pred)
    true_negatives = rf_matrix[0][0]
    false_negatives = rf_matrix[1][0]
    true_positives = rf_matrix[1][1]
    false_positives = rf_matrix[0][1]
    accuracy = (true_negatives + true_positives) / (true_negatives + true_positives + false_negatives + false_positives)
    percision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    specificity = true_negatives / (true_negatives + false_positives)
    print('Accuracy: {}'.format(float(accuracy)))
    print('Percision: {}'.format(float(percision)))
    print('Recall: {}'.format(float(recall)))
    print('Specificity: {}'.format(float(specificity)))
#time
run_randomForest(X_train_reg, X_test_reg, y_train, y_test)
#time
run_randomForest(X_train, X_test, y_train, y_test)
#print(X_train.shape)

sel = SelectFromModel(LogisticRegression(penalty = 'l1', C = 0.05, solver = 'liblinear'))
sel.fit(X_train, y_train)
#print(sel.get_support())

#print(sel.estimator_.coef_)

X_train_l1 = sel.transform(X_train)
X_test_l1 = sel.transform(X_test)
#time
run_randomForest(X_train_l1, X_test_l1, y_train, y_test)

sel = SelectFromModel(LogisticRegression(penalty = 'l2', C = 0.05, solver = 'liblinear'))
sel.fit(X_train, y_train)
#print(sel.get_support())

#print(sel.estimator_.coef_)

X_train_l2 = sel.transform(X_train)
X_test_l2 = sel.transform(X_test)
#time
(run_randomForest(X_train_l2, X_test_l2, y_train, y_test))





