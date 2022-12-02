import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix , classification_report
from sklearn.model_selection import train_test_split
from requests.packages import target
from datetime import datetime
# Bagged Trees Regressor
from sklearn.ensemble import BaggingRegressor
import warnings
warnings.filterwarnings('ignore')
start = datetime.now()
df = pd.read_csv("C:/Users/as097/Desktop/PVC/tsfelWRM_multi.csv")
#print(df.sample(5))
#print(df.Label.value_counts())


df.drop('0_ECDF Percentile Count_0',axis='columns',inplace=True)
#print(df.dtypes)

X = df.drop('Label1',axis='columns')
y = testLabels = df.Label1.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=15, stratify=y)
# training a Naive Bayes classifier
#from memory_profiler import profile


# instantiating the decorator
@profile
def NB(X_train, y_train, X_test, y_test):
    from sklearn.naive_bayes import GaussianNB

    gnb = GaussianNB().fit(X_train, y_train)
    y_preds = gnb.predict(X_test)
    # accuracy on X_test
    accuracy = gnb.score(X_test, y_test)
    print(accuracy)

    print("Classification Report: \n", classification_report(y_test, y_preds))
    return y_preds

y_preds = NB(X_train, y_train, X_test, y_test)


# creating a confusion matrix
cm = confusion_matrix(y_test, y_preds)
print(cm)

# now we have initialized the variable
# end to store the ending time after
# execution of program
end = datetime.now()

# difference of start and end variables
# gives the time of execution of the
# program in between
print("The time of execution of above program is :",
      str(end-start)[5:])

'''
if __name__ == '__main__':
    ANN(X_train, y_train, X_test, y_test)
'''


import time
import multiprocessing as mp
import psutil
import numpy as np

def monitor(target):
    worker_process = mp.Process(target=target)
    worker_process.start()
    p = psutil.Process(worker_process.pid)

    # log cpu usage of `worker_process` every 10 ms
    cpu_percents = []
    while worker_process.is_alive():
        cpu_percents.append(p.cpu_percent())
        time.sleep(0.01)

    worker_process.join()
    return cpu_percents
if __name__ ==  '__main__':
    cpu_percents = monitor(target)
    plt.plot(cpu_percents)
    plt.show()
