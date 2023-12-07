import pandas as pd

# 讀取CSV文件
file_path = 'testing_data.csv'
data = pd.read_csv(file_path)

# # 保留 label 欄位
# labels = data['label']
# data = data.drop('label', axis=1)

# 指定需要檢查的欄位
columns_to_check = ['x0', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14']

# 对于标称变量x1，使用众数填充
x1_mode = data['x1'].mode()[0]
data['x1'].fillna(x1_mode, inplace=True)

# 創建一個新的 DataFrame 來保存不同填補方式的資料
data_filled_mean = data.copy()

# 使用平均值填補缺失值
data_filled_mean[columns_to_check] = data_filled_mean[columns_to_check].apply(lambda x: x.fillna(x.mean()))

# data_filled_mean['label'] = labels

# 將結果保存為新的 CSV 文件
output_file_path_mean = 'testing_data_filled_mean.csv'

data_filled_mean.to_csv(output_file_path_mean, index=False)

# 顯示成功訊息
print(f"以平均值填補缺失值的資料已保存至 {output_file_path_mean}")
