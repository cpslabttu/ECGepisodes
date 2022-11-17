import tsfel
import zipfile
import numpy as np
import pandas as pd


dir=("C:/Users/as097/Desktop/PVC/")
ecg_signal=pd.read_csv(dir+'chf10.csv')['ECG']

# Store the dataset as a Pandas dataframe.

X_train_sig = pd.DataFrame(np.hstack(ecg_signal), columns=["total_acc_x"])

#print (X_train_sig)

cfg_file = tsfel.get_features_by_domain()                                                      # If no argument is passed retrieves all available features
X_train = tsfel.time_series_features_extractor(cfg_file, X_train_sig, fs=128, window_size=50)    # Receives a time series sampled at 50 Hz, divides into windows of size 250 (i.e. 5 seconds) and extracts all features


cfg_file = tsfel.get_features_by_domain()               # All features will be extracted.
cgf_file = tsfel.get_features_by_domain("statistical")  # All statistical domain features will be extracted
cgf_file = tsfel.get_features_by_domain("temporal")     # All temporal domain features will be extracted
cgf_file = tsfel.get_features_by_domain("spectral")     # All spectral domain features will be extracted
# save in file
path = 'C:/Users/as097/Desktop/'
X_train.to_csv(path + 'tsfelFeatures.csv')