
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix , classification_report



df = pd.read_csv("C:/Users/as097/Desktop/PVC/tsfelWRM_multi.csv")
df.drop('0_ECDF Percentile Count_0', axis='columns', inplace=True)
X = df.drop('Label',axis='columns')
y = testLabels = df.Label1.astype(np.float32)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=15, stratify=y)
# training a DescisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

dtree_model = DecisionTreeClassifier(max_depth=40).fit(X_train, y_train)
y_preds = dtree_model.predict(X_test)
print("Classification Report: \n", classification_report(y_test, y_preds))


# creating a confusion matrix
cm = confusion_matrix(y_test, y_preds)
print(cm)

