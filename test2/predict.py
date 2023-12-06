import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 讀取訓練數據
file_path = 'data_resampled.csv'
data = pd.read_csv(file_path)

# 提取特徵和標籤
X = data[['col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9', 'col10', 'col11', 'col12', 'col13']]
y = data['label']

# 拆分數據集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特徵標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 構建深度學習模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(12,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 編譯模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 訓練模型
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.2)

# 評估模型
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
print(f'Test Accuracy: {test_accuracy}')

# 讀取預測數據
prediction_data_path = 'testing_data.csv'
prediction_data = pd.read_csv(prediction_data_path)

# 提取特徵
X_pred = prediction_data[['col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9', 'col10', 'col11', 'col12', 'col13']]

# 特徵標準化
X_pred_scaled = scaler.transform(X_pred)

# 預測
predictions = model.predict(X_pred_scaled)
# 將預測結果添加到預測數據中
prediction_data['label'] = np.round(predictions).astype(int)

# 保留 'id' 和 'predicted_label' 列
result_df = prediction_data[['id', 'label']]

# 保存包含預測結果的 CSV 文件
result_df.to_csv('predicted_results.csv', index=False)
