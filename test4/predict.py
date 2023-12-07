import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from tensorflow import keras
from tensorflow.keras import layers

# 讀取訓練數據
data = pd.read_csv("data_resampled.csv")

# 分割特徵和標籤
X = data.drop(columns=['id', 'label'])
y = data['label']

# 對標籤進行編碼
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 標準化特徵
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# 使用 SelectKBest 進行特徵選擇
k_best = 15  # 設定保留的特徵數量
selector = SelectKBest(f_classif, k=k_best)

# 適應特徵選擇器到訓練數據
X_train_selected = selector.fit_transform(X_train, y_train)

# 使用相同的特徵選擇器轉換測試數據
X_test_selected = selector.transform(X_test)

# 獲取選擇的屬性名稱
selected_features = X.columns[selector.get_support()]

# 輸出選擇的屬性
print("選擇的屬性：", selected_features)

# 建立深度學習模型
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_selected.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(len(label_encoder.classes_), activation='softmax')
])

# 編譯模型
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 訓練模型
model.fit(X_train_selected, y_train, epochs=10, batch_size=32, validation_split=0.2)

# 在測試集上評估模型
y_pred = model.predict(X_test_selected)
y_pred_classes = [label.argmax() for label in y_pred]
accuracy = accuracy_score(y_test, y_pred_classes)
print(f"準確率: {accuracy}")

# 讀取測試數據
test_data = pd.read_csv("testing_data_filled_mean.csv")

# 對x1進行編碼
x1_encoder = LabelEncoder()
x1_encoded = x1_encoder.fit_transform(test_data['x1'])
test_data['x1'] = x1_encoded

# 使用相同的特徵選擇器轉換測試數據
X_test_final = selector.transform(test_data.drop(columns=['id']))

# 進行預測
y_pred = model.predict(X_test_final)

# 將預測轉換為原始標籤格式
predicted_labels = label_encoder.inverse_transform([label.argmax() for label in y_pred])

# 將預測結果加入原始測試數據
test_data['label'] = predicted_labels

# 重新編號 id 從 0 開始
test_data['id'] = range(len(test_data))

# 保存包含預測結果的 CSV 文件
test_data[['id', 'label']].to_csv("predicted_results.csv", index=False)
