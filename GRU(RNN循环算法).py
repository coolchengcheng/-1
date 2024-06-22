import numpy as np
import pandas as pd
import glob
from sklearn.preprocessing import MinMaxScaler
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Function to load and merge CSV files
def load_and_merge_files(file_pattern):
    all_files = glob.glob(file_pattern)
    if not all_files:
        raise FileNotFoundError(f"No files found for pattern: {file_pattern}")
    df_list = [pd.read_csv(file) for file in all_files]
    combined_df = pd.concat(df_list)
    return combined_df

# Load data
file_pattern = 'PRSA_Data_20130301-20170228/*.csv'
data = load_and_merge_files(file_pattern)

# Preprocess data
data['date'] = pd.to_datetime(data[['year', 'month', 'day']])
data.set_index('date', inplace=True)
data = data.sort_index()

features = ['PM2.5', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
data = data[features]
data = data.fillna(method='ffill').fillna(method='bfill')

train_data = data['2013-03-01':'2016-11-30']
test_data = data['2016-12-01':'2017-02-28']

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(train_data)
scaled_test_data = scaler.transform(test_data)

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        X.append(dataset[i:(i+look_back), :])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 3
X_train, Y_train = create_dataset(scaled_train_data, look_back)
X_test, Y_test = create_dataset(scaled_test_data, look_back)

X_train = np.reshape(X_train, (X_train.shape[0], look_back, X_train.shape[2]))
X_test = np.reshape(X_test, (X_test.shape[0], look_back, X_test.shape[2]))

# Build and train GRU model
model = Sequential()
model.add(GRU(50, input_shape=(look_back, X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(50))
model.add(Dense(1))

model.compile(optimizer=Adam(), loss='mean_squared_error')

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(X_train, Y_train, epochs=50, batch_size=64, validation_split=0.1, verbose=2, callbacks=[early_stopping])

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(np.concatenate((train_predict, np.zeros((train_predict.shape[0], len(features)-1))), axis=1))[:,0]
test_predict = scaler.inverse_transform(np.concatenate((test_predict, np.zeros((test_predict.shape[0], len(features)-1))), axis=1))[:,0]

train_Y = scaler.inverse_transform(np.concatenate((Y_train.reshape(-1, 1), np.zeros((Y_train.shape[0], len(features)-1))), axis=1))[:,0]
test_Y = scaler.inverse_transform(np.concatenate((Y_test.reshape(-1, 1), np.zeros((Y_test.shape[0], len(features)-1))), axis=1))[:,0]

plt.figure(figsize=(12, 6))
plt.plot(data.index, data['PM2.5'], label='Actual PM2.5')
plt.plot(train_data.index[look_back+1:], train_predict, label='Train Predict PM2.5')
plt.plot(test_data.index[look_back+1:], test_predict, label='Test Predict PM2.5')
plt.xlabel('Date')
plt.ylabel('PM2.5 Concentration')
plt.legend()
plt.show()
