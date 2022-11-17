import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
#%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif, f_regression

data = pd.read_csv('C:/Users/as097/Desktop/PVC/tsfelFeatures.csv', nrows = 20000)
#print(data.head())

#data = data[['0_FFT mean coefficient_22', 'Label']].copy()
#data = data[['0_FFT mean coefficient_22','0_Power bandwidth', 'Label']].copy()
#data = data[['0_FFT mean coefficient_22','0_Power bandwidth','0_Histogram_3', 'Label']].copy()
#data = data[['0_FFT mean coefficient_22','0_Power bandwidth','0_Histogram_3', '0_Signal distance', '0_ECDF Percentile Count_1', '0_ECDF_8', '0_Histogram_5', '0_Root mean square', '0_FFT mean coefficient_23', '0_Autocorrelation', 'Label']].copy()
#X = data.copy()
X = data.drop('Label', axis = 1)
y = data['Label']

#print(X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)

#remove constant and quasi constant features
constant_filter = VarianceThreshold(threshold=0.01)
constant_filter.fit(X_train)
X_train_filter = constant_filter.transform(X_train)
X_test_filter = constant_filter.transform(X_test)
#print(X_train_filter.shape, X_test_filter.shape)

#remove duplicate features
X_train_T = X_train_filter.T
X_test_T = X_test_filter.T
X_train_T = pd.DataFrame(X_train_T)
X_test_T = pd.DataFrame(X_test_T)
#print(X_train_T.duplicated().sum())

duplicated_features = X_train_T.duplicated()
features_to_keep = [not index for index in duplicated_features]

X_train_unique = X_train_T[features_to_keep].T
X_test_unique = X_test_T[features_to_keep].T
scaler = StandardScaler().fit(X_train_unique)
X_train_unique = scaler.transform(X_train_unique)
X_test_unique = scaler.transform(X_test_unique)
X_train_unique = pd.DataFrame(X_train_unique)
X_test_unique = pd.DataFrame(X_test_unique)
#print(X_train_unique.shape, X_test_unique.shape)

corrmat = X_train_unique.corr()
#find correlated features
def get_correlation(data, threshold):
    corr_col = set()
    corrmat = data.corr()
    for i in range(len(corrmat.columns)):
        for j in range(i):
            if abs(corrmat.iloc[i, j]) > threshold:
                colname = corrmat.columns[i]
                corr_col.add(colname)
    return corr_col

corr_features = get_correlation(X_train_unique, 0.70)
#print('correlated features: ', len(set(corr_features)) )
X_train_uncorr = X_train_unique.drop(labels=corr_features, axis = 1)
X_test_uncorr = X_test_unique.drop(labels = corr_features, axis = 1)
#print(X_train_uncorr.shape, X_test_uncorr.shape)

# Feature Ranking
'''
se = f_classif(X_train_uncorr, y_train)
p_values = pd.Series(se[1])
p_values.index = X_train_uncorr.columns
p_values.sort_values(ascending = True, inplace = True)
p_values.plot.bar(figsize = (16, 5))
plt.show()

se = f_classif(X_train_uncorr, y_train)
p_values = pd.Series(se[1], index = X_train_uncorr.columns)
p_values.sort_values(ascending=False, inplace = True)
p_values.plot.bar()
plt.show()
'''

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=1)
X_train_lda = lda.fit_transform(X_train_uncorr, y_train)
X_test_lda = lda.transform(X_test_uncorr)
#print(X_train_lda.shape, X_test_lda.shape)

def run_randomForest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Accuracy on test set: ')
    print(accuracy_score(y_test, y_pred))
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
run_randomForest(X_train_lda, X_test_lda, y_train, y_test)
#time
run_randomForest(X_train_uncorr, X_test_uncorr, y_train, y_test)
#time
run_randomForest(X_train, X_test, y_train, y_test)

from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=42)
pca.fit(X_train_uncorr)
X_train_pca = pca.transform(X_train_uncorr)
X_test_pca = pca.transform(X_test_uncorr)
print(X_train_pca.shape, X_test_pca.shape)

#time
run_randomForest(X_train_pca, X_test_pca, y_train, y_test)
#time
run_randomForest(X_train, X_test, y_train, y_test)
print(X_train_uncorr.shape)

for component in range(1,30):
    pca = PCA(n_components=component, random_state=42)
    pca.fit(X_train_uncorr)
    X_train_pca = pca.transform(X_train_uncorr)
    X_test_pca = pca.transform(X_test_uncorr)
    print('Selected Components: ', component)
    run_randomForest(X_train_pca, X_test_pca, y_train, y_test)
    print()





