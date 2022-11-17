import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

data = pd.read_csv('C:/Users/as097/Desktop/tsfelFeatures.csv')
'''
originalFeatures = data.columns
print('originalFeatures count', len(originalFeatures))
print('originalFeatures', originalFeatures)
print(data.head())
alpha = 0.02
plt.figure(figsize=(10,10))
# 0_ECDF_0 and 0_ECDF_1
plt.subplots(121)
plt.scatter(data.Wavelet_variance_0, data.Wavelet_variance_1, color='blue', alpha=alpha)
plt.title('0_ECDF_0 and 0_ECDF_1')
plt.savefig('0_ECDF_0 and 0_ECDF_1.png')
'''

X = data.iloc[:,0:20]  #independent columns
y = data.iloc[:,-1]    #target column i.e price range
#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.Wavelet_variance_0)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
featureScores