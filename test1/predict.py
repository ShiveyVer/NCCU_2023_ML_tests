import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 讀取資料
data = pd.read_csv('training_data.csv')  # 將'your_data.csv'替換為你的資料檔案路徑

# 切分資料集為訓練集和測試集
X = data.drop('label', axis=1)
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立隨機森林分類器
clf = RandomForestClassifier(random_state=42)

# 訓練模型
clf.fit(X_train, y_train)

# 讀取測試資料
testing_data = pd.read_csv('testing_data.csv')  # 將'testing_data.csv'替換為你的測試資料檔案路徑

# 預測測試集
y_pred = clf.predict(testing_data)

# 輸出預測結果 (僅包含 'id' 和 'label' 欄位)
predictions = pd.DataFrame({'id': testing_data['id'], 'label': y_pred})
predictions.to_csv('predicted_results.csv', index=False)

print('Predictions saved to predicted_results.csv')