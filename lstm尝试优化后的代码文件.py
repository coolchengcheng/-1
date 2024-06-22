import numpy as np
import pandas as pd
import glob
from sklearn.preprocessing import MinMaxScaler
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# 定义函数 load_and_merge_files，用于加载并合并所有符合文件模式的CSV文件
def load_and_merge_files(file_pattern):
    all_files = glob.glob(file_pattern)
    if not all_files:
        raise FileNotFoundError(f"No files found for pattern: {file_pattern}")
    df_list = [pd.read_csv(file) for file in all_files]
    combined_df = pd.concat(df_list)
    return combined_df

# 设置文件路径模式并调用 load_and_merge_files 函数加载数据
file_pattern = 'PRSA_Data_20130301-20170228/*.csv'
data = load_and_merge_files(file_pattern)

# 假设文件中有 year, month, day 列
data['date'] = pd.to_datetime(data[['year', 'month', 'day']])
# 将 date 列设为索引
data.set_index('date', inplace=True)
# 确保索引是按时间顺序排序的
data = data.sort_index()

# 增加气象数据特征（假设数据中已有这些列）
features = ['PM2.5', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
additional_features = ['NO2', 'SO2', 'CO']
data = data[features]

# 处理缺失值，使用前向填充和后向填充方法
data = data.fillna(method='ffill').fillna(method='bfill')

# 分割训练集和测试集
train_data = data['2013-03-01':'2016-11-30']
test_data = data['2016-12-01':'2017-02-28']

# 使用 MinMaxScaler 进行数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(train_data)
scaled_test_data = scaler.transform(test_data)

# 定义函数 create_dataset，根据给定的 look_back 参数创建训练和测试数据集
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        X.append(dataset[i:(i+look_back), :])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# 设置 look_back 参数，并创建训练集和测试集
look_back = 3
X_train, Y_train = create_dataset(scaled_train_data, look_back)
X_test, Y_test = create_dataset(scaled_test_data, look_back)

# 重塑输入数据为LSTM需要的格式 [样本数, 时间步长, 特征数]
X_train = np.reshape(X_train, (X_train.shape[0], look_back, X_train.shape[2]))
X_test = np.reshape(X_test, (X_test.shape[0], look_back, X_test.shape[2]))

# 定义性能评估函数
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_performance(true_values, predictions):
    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predictions)
    return mse, rmse, mae

# 训练和评估LSTM模型
lstm_model = Sequential()
lstm_model.add(LSTM(50, input_shape=(look_back, X_train.shape[2]), return_sequences=True))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(50))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer=Adam(), loss='mean_squared_error')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lstm_model.fit(X_train, Y_train, epochs=50, batch_size=64, validation_split=0.1, verbose=2, callbacks=[early_stopping])

lstm_train_predict = lstm_model.predict(X_train)
lstm_test_predict = lstm_model.predict(X_test)

lstm_train_predict = scaler.inverse_transform(np.concatenate((lstm_train_predict, np.zeros((lstm_train_predict.shape[0], len(features)-1))), axis=1))[:,0]
lstm_test_predict = scaler.inverse_transform(np.concatenate((lstm_test_predict, np.zeros((lstm_test_predict.shape[0], len(features)-1))), axis=1))[:,0]

train_Y = scaler.inverse_transform(np.concatenate((Y_train.reshape(-1, 1), np.zeros((Y_train.shape[0], len(features)-1))), axis=1))[:,0]
test_Y = scaler.inverse_transform(np.concatenate((Y_test.reshape(-1, 1), np.zeros((Y_test.shape[0], len(features)-1))), axis=1))[:,0]

lstm_train_mse, lstm_train_rmse, lstm_train_mae = evaluate_performance(train_Y, lstm_train_predict)
lstm_test_mse, lstm_test_rmse, lstm_test_mae = evaluate_performance(test_Y, lstm_test_predict)

print("LSTM Model Performance:")
print(f"Train MSE: {lstm_train_mse}, Train RMSE: {lstm_train_rmse}, Train MAE: {lstm_train_mae}")
print(f"Test MSE: {lstm_test_mse}, Test RMSE: {lstm_test_rmse}, Test MAE: {lstm_test_mae}")

# 训练和评估GRU模型
gru_model = Sequential()
gru_model.add(GRU(50, input_shape=(look_back, X_train.shape[2]), return_sequences=True))
gru_model.add(Dropout(0.2))
gru_model.add(GRU(50))
gru_model.add(Dense(1))
gru_model.compile(optimizer=Adam(), loss='mean_squared_error')
gru_model.fit(X_train, Y_train, epochs=50, batch_size=64, validation_split=0.1, verbose=2, callbacks=[early_stopping])

gru_train_predict = gru_model.predict(X_train)
gru_test_predict = gru_model.predict(X_test)

gru_train_predict = scaler.inverse_transform(np.concatenate((gru_train_predict, np.zeros((gru_train_predict.shape[0], len(features)-1))), axis=1))[:,0]
gru_test_predict = scaler.inverse_transform(np.concatenate((gru_test_predict, np.zeros((gru_test_predict.shape[0], len(features)-1))), axis=1))[:,0]

gru_train_mse, gru_train_rmse, gru_train_mae = evaluate_performance(train_Y, gru_train_predict)
gru_test_mse, gru_test_rmse, gru_test_mae = evaluate_performance(test_Y, gru_test_predict)

print("GRU Model Performance:")
print(f"Train MSE: {gru_train_mse}, Train RMSE: {gru_train_rmse}, Train MAE: {gru_train_mae}")
print(f"Test MSE: {gru_test_mse}, Test RMSE: {gru_test_rmse}, Test MAE: {gru_test_mae}")

# 训练和评估CNN模型
cnn_model = Sequential()
cnn_model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(look_back, X_train.shape[2])))
cnn_model.add(Dropout(0.2))
cnn_model.add(MaxPooling1D(pool_size=2))
cnn_model.add(Flatten())
cnn_model.add(Dense(50, activation='relu'))
cnn_model.add(Dense(1))
cnn_model.compile(optimizer=Adam(), loss='mean_squared_error')
cnn_model.fit(X_train, Y_train, epochs=50, batch_size=64, validation_split=0.1, verbose=2, callbacks=[early_stopping])

cnn_train_predict = cnn_model.predict(X_train)
cnn_test_predict = cnn_model.predict(X_test)

cnn_train_predict = scaler.inverse_transform(np.concatenate((cnn_train_predict, np.zeros((cnn_train_predict.shape[0], len(features)-1))), axis=1))[:,0]
cnn_test_predict = scaler.inverse_transform(np.concatenate((cnn_test_predict, np.zeros((cnn_test_predict.shape[0], len(features)-1))), axis=1))[:,0]

cnn_train_mse, cnn_train_rmse, cnn_train_mae = evaluate_performance(train_Y, cnn_train_predict)
cnn_test_mse, cnn_test_rmse, cnn_test_mae = evaluate_performance(test_Y, cnn_test_predict)

print("CNN Model Performance:")
print(f"Train MSE: {cnn_train_mse}, Train RMSE: {cnn_train_rmse}, Train MAE: {cnn_train_mae}")
print(f"Test MSE: {cnn_test_mse}, Test RMSE: {cnn_test_rmse}, Test MAE: {cnn_test_mae}")

# 总结和比较三个模型的性能
models_performance = {
    'LSTM': {'Train MSE': lstm_train_mse, 'Train RMSE': lstm_train_rmse, 'Train MAE': lstm_train_mae,
             'Test MSE': lstm_test_mse, 'Test RMSE': lstm_test_rmse, 'Test MAE': lstm_test_mae},
    'GRU': {'Train MSE': gru_train_mse, 'Train RMSE': gru_train_rmse, 'Train MAE': gru_train_mae,
            'Test MSE': gru_test_mse, 'Test RMSE': gru_test_rmse, 'Test MAE': gru_test_mae},
    'CNN': {'Train MSE': cnn_train_mse, 'Train RMSE': cnn_train_rmse, 'Train MAE': cnn_train_mae,
            'Test MSE': cnn_test_mse, 'Test RMSE': cnn_test_rmse, 'Test MAE': cnn_test_mae}
}

for model_name, performance in models_performance.items():
    print(f"{model_name} Model Performance:")
    for metric, value in performance.items():
        print(f"{metric}: {value}")
    print()

# 可视化结果
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['PM2.5'], label='Actual PM2.5')
plt.plot(train_data.index[look_back+1:], lstm_train_predict, label='LSTM Train Predict PM2.5')
plt.plot(test_data.index[look_back+1:], lstm_test_predict, label='LSTM Test Predict PM2.5')
plt.plot(train_data.index[look_back+1:], gru_train_predict, label='GRU Train Predict PM2.5')
plt.plot(test_data.index[look_back+1:], gru_test_predict, label='GRU Test Predict PM2.5')
plt.plot(train_data.index[look_back+1:], cnn_train_predict, label='CNN Train Predict PM2.5')
plt.plot(test_data.index[look_back+1:], cnn_test_predict, label='CNN Test Predict PM2.5')
plt.xlabel('Date')
plt.ylabel('PM2.5 Concentration')
plt.legend()
plt.show()
