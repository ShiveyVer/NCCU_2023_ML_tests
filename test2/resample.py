import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# 讀取原始數據
file_path = 'data_filled_mean.csv'
data = pd.read_csv(file_path)

# 分離特徵和標籤
X = data.drop('label', axis=1)
y = data['label']

# 將數據集分為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用SMOTE過採樣
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 將 resample 後的數據組合成 DataFrame
resampled_data = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name='label')], axis=1)

# 將 resample 後的數據保存為 CSV 文件
output_file_path = 'data_resampled.csv'
resampled_data.to_csv(output_file_path, index=False)

# 顯示成功訊息
print(f"Resampled 數據已保存至 {output_file_path}")
