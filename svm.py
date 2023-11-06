from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Đường dẫn đến tệp dữ liệu
csv_file_path = "C:\\Users\\FPT\Desktop\\BTL_Nhom3"

# Đọc dữ liệu từ tệp CSV vào DataFrame
data = pd.read_csv(csv_file_path)

# Chia dữ liệu thành các đặc trưng (features) và nhãn (labels)
X = data[['Gender', 'Height', 'Weight']]
y = data['Index']

# Mã hóa one-hot cho thuộc tính "Gender"
X_encoded = pd.get_dummies(X, columns=['Gender'], drop_first=False)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Khởi tạo mô hình SVM
svm_model = SVC()

# Huấn luyện mô hình trên tập huấn luyện
svm_model.fit(X_train, y_train)

# Dự đoán giá trị cho tập kiểm tra
y_pred_test_svm = svm_model.predict(X_test)

# In ra dự đoán giá trị cho tập kiểm tra
print("Du doan gia tri cho tap kiem tra:")
print(y_pred_test_svm)

# So sánh kết quả dự đoán và kết quả thực tế
comparison_df_svm = pd.DataFrame({'Thuc te': y_test, 'Du doan': y_pred_test_svm})
print("\nSo sanh ket qua du doan va ket qua thuc te:")
print(comparison_df_svm)

# Đánh giá hiệu suất của mô hình bằng các độ đo phù hợp
accuracy = accuracy_score(y_test, y_pred_test_svm)
precision = precision_score(y_test, y_pred_test_svm, average='weighted')
recall = recall_score(y_test, y_pred_test_svm, average='weighted')
f1 = f1_score(y_test, y_pred_test_svm, average='weighted')

print("\nDo chinh xac (Accuracy):", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)