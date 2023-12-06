import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 讀取數據
file_path = 'training_data.csv'
data = pd.read_csv(file_path)

# 指定需要填補的欄位
columns_to_impute = ['col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9', 'col10', 'col11', 'col12', 'col13']

# 分離具有缺失值和非缺失值的樣本
missing_data = data[data[columns_to_impute].isnull().any(axis=1)]
non_missing_data = data.dropna(subset=columns_to_impute)

# 初始化隨機森林模型
models = {column: RandomForestRegressor() for column in columns_to_impute}

# 訓練模型
for column in columns_to_impute:
    X_train, X_test, y_train, y_test = train_test_split(
        non_missing_data.drop(columns=columns_to_impute),
        non_missing_data[column],
        test_size=0.2,
        random_state=42
    )
    models[column].fit(X_train, y_train)

# 預測缺失值
for column in columns_to_impute:
    predicted_values = models[column].predict(missing_data.drop(columns=columns_to_impute))
    data.loc[data[column].isnull(), column] = predicted_values

# 將填補後的數據保存為新的 CSV 文件
output_file_path = 'output_filled_rf.csv'
data.to_csv(output_file_path, index=False)

# 查看填補後的數據統計信息
print("填補後的數據統計信息：")
print(data.describe())

# 顯示成功訊息
print(f"填補後的數據已保存至 {output_file_path}")
