import pandas as pd
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder

# 讀取CSV文件
input_csv = 'training_data_filled_mean.csv'
output_csv = 'training_data_no_outliner.csv'
data = pd.read_csv(input_csv)

# 對x1進行編碼
x1_encoder = LabelEncoder()
x1_encoded = x1_encoder.fit_transform(data['x1'])
data['x1'] = x1_encoded

# 選擇要檢測和剔除離群值的特徵列
features = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14']

# 計算z分數
z_scores = zscore(data[features])

# 設定離群值的閾值，例如3或-3
threshold = 3

# 找到超過閾值的索引
outliers = (z_scores > threshold).any(axis=1)

# 剔除離群值
data_no_outliers = data[~outliers]


# 將預測轉換為原始標籤格式
x1s = x1_encoder.inverse_transform([label.argmax() for label in data])

# 將預測結果加入原始測試數據
data['x1'] = x1s

# 將處理過的數據保存到新的CSV文件
data_no_outliers.to_csv(output_csv, index=False)
