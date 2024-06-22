import numpy as np
import pandas as pd
import glob
from sklearn.preprocessing import MinMaxScaler
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os

# 定义函数 load_and_merge_files，用于加载并合并所有符合文件模式的CSV文件
def load_and_merge_files(file_pattern):
    # 使用 glob.glob 找到所有符合模式的文件路径名
    all_files = glob.glob(file_pattern)
    # 如果没有找到文件，抛出一个 FileNotFoundError 异常
    if not all_files:
        raise FileNotFoundError(f"No files found for pattern: {file_pattern}")
    df_list = []
    # 对于每个找到的文件，读取CSV文件并将其添加到 df_list 中
    for file in all_files:
        df = pd.read_csv(file)
        df_list.append(df)
    # 使用 pd.concat 将所有DataFrame合并成一个DataFrame并返回
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

# 创建和训练改进的LSTM模型
model = Sequential()
# 添加第一层LSTM，并设置返回序列
model.add(LSTM(50, input_shape=(look_back, X_train.shape[2]), return_sequences=True))
# 添加 Dropout 层，防止过拟合
model.add(Dropout(0.2))
# 添加第二层LSTM
model.add(LSTM(50))
# 添加输出层
model.add(Dense(1))


# 编译模型，设置优化器和损失函数
model.compile(optimizer=Adam(), loss='mean_squared_error')

# 使用EarlyStopping来防止过拟合并加快训练
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 训练模型，设置批量大小和训练轮数
model.fit(X_train, Y_train, epochs=50, batch_size=64, validation_split=0.1, verbose=2, callbacks=[early_stopping])

# 使用训练好的模型对训练集和测试集进行预测
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 反归一化预测值
train_predict = scaler.inverse_transform(np.concatenate((train_predict, np.zeros((train_predict.shape[0], len(features)-1))), axis=1))[:,0]
test_predict = scaler.inverse_transform(np.concatenate((test_predict, np.zeros((test_predict.shape[0], len(features)-1))), axis=1))[:,0]

# 反归一化实际值
train_Y = scaler.inverse_transform(np.concatenate((Y_train.reshape(-1, 1), np.zeros((Y_train.shape[0], len(features)-1))), axis=1))[:,0]
test_Y = scaler.inverse_transform(np.concatenate((Y_test.reshape(-1, 1), np.zeros((Y_test.shape[0], len(features)-1))), axis=1))[:,0]

# 可视化结果
plt.figure(figsize=(12, 6))
# 绘制实际PM2.5浓度
plt.plot(data.index, data['PM2.5'], label='Actual PM2.5')
# 绘制训练集上的预测值
plt.plot(train_data.index[look_back+1:], train_predict, label='Train Predict PM2.5')
# 绘制测试集上的预测值
plt.plot(test_data.index[look_back+1:], test_predict, label='Test Predict PM2.5')
plt.xlabel('Date')
plt.ylabel('PM2.5 Concentration')
plt.legend()
plt.show()
