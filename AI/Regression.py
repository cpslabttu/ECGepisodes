
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


df = pd.read_csv('C:/Users/as097/Desktop/PVC/tsfelFeatures.csv')
#X = pd.get_dummies(df.drop(['0_ECDF_6'], axis=1))
#y = df['Label'].apply(lambda x: 1 if x=='Yes' else 0)
df = pd.get_dummies(df, prefix='', prefix_sep='')
X = df.sample(frac=0.8, random_state=0)
Y = X.drop(X.index)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
#print(df.head())
sns.pairplot(X[['0_Absolute energy', '0_Area under the curve', '0_Centroid', '0_ECDF Percentile_0', '0_Entropy', '0_FFT mean coefficient_1']], diag_kind='kde')
plt.savefig('1.png')
#print(X.describe().transpose())

train_features = X.copy()
test_features = Y.copy()

train_labels = train_features.pop('Label')
test_labels = test_features.pop('Label')

#print(X.describe().transpose()[['mean', 'std']])

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))
#print(normalizer.mean.numpy())

first = np.array(train_features[:1])

with np.printoptions(precision=2, suppress=True):
  #print('First example:', first)
  #print()
  #print('Normalized:', normalizer(first).numpy())

    abs = np.array(train_features['0_Absolute energy'])
    abs_normalizer = layers.Normalization(input_shape=[1, ], axis=None)
    abs_normalizer.adapt(abs)
    abs_model = tf.keras.Sequential([
        abs_normalizer,
        layers.Dense(units=1)
    ])
    #print(abs_model.summary())
    #print(abs_model.predict(abs[:10]))

    abs_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
        loss='mean_absolute_error')
    # time
    history = abs_model.fit(
        train_features['0_Absolute energy'],
        train_labels,
        epochs=100,
        # Suppress logging.
        verbose=0,
        # Calculate validation results on 20% of the training data.
        validation_split=0.2)
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist.tail())
test_results = {}

test_results['abs_model'] = abs_model.evaluate(
    test_features['0_Absolute energy'],
    test_labels, verbose=0)
x = tf.linspace(0.0, 250, 251)
y = abs_model.predict(x)
def plot_abs(x, y):
  plt.scatter(train_features['0_Absolute energy'], train_labels, label='Data')
  plt.plot(x, y, color='k', label='Predictions')
  plt.xlabel('Abs')
  plt.ylabel('Label')
  plt.legend()
  plot_abs(x, y)
  plt.savefig('2.png')

'''
model = Sequential()
model.add(Dense(units=32, activation='relu', input_dim=len(X_train.columns)))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics='accuracy')
model.fit(X_train, y_train, epochs=10, batch_size=32)
y_hat = model.predict(X_test)
y_hat = [0 if val < 0.5 else 1 for val in y_hat]
print(accuracy_score(y_test, y_hat))
'''