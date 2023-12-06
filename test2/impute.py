import pandas as pd

# 讀取CSV文件
file_path = 'training_data.csv'
data = pd.read_csv(file_path)

# 保留 label 欄位
labels = data['label']
data = data.drop('label', axis=1)

# 指定需要檢查的欄位
columns_to_check = ['col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9', 'col10', 'col11', 'col12', 'col13']

# 創建一個新的 DataFrame 來保存不同填補方式的資料
data_filled_mean = data.copy()

# 使用平均值填補缺失值
data_filled_mean[columns_to_check] = data_filled_mean[columns_to_check].apply(lambda x: x.fillna(x.mean()))

data_filled_mean['label'] = labels

# 將結果保存為新的 CSV 文件
output_file_path_mean = 'data_filled_mean.csv'

data_filled_mean.to_csv(output_file_path_mean, index=False)

# 顯示成功訊息
print(f"以平均值填補缺失值的資料已保存至 {output_file_path_mean}")
