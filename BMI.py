from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Đường dẫn đến tệp dữ liệu
csv_file_path = "C:\\Users\\FPT\\Desktop\\archive\\bmi.csv"

# Đọc dữ liệu từ tệp CSV vào DataFrame
data = pd.read_csv(csv_file_path)

# Chia dữ liệu thành các đặc trưng (features) và nhãn (labels)
X = data[['Gender', 'Height', 'Weight']]
y = data['Index']

# Mã hóa one-hot cho thuộc tính "Gender"
X_encoded = pd.get_dummies(X, columns=['Gender'], drop_first=False)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Hiển thị X_train và y_train
print("X_train:")
print(X_train)
print("y_train:")
print(y_train)
print("y_test:")
print(y_test)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Khởi tạo mô hình hồi quy tuyến tính
linear_model = LinearRegression()

# Huấn luyện mô hình trên tập huấn luyện
linear_model.fit(X_train, y_train)

# Dự đoán giá trị cho tập kiểm tra
y_pred_test = linear_model.predict(X_test)

# In ra dự đoán giá trị cho tập kiểm tra
print("Dự đoán giá trị cho tập kiểm tra:")
print(y_pred_test)

# So sánh kết quả dự đoán và kết quả thực tế
comparison_df = pd.DataFrame({'Thực tế': y_test, 'Dự đoán': y_pred_test})
print("\nSo sánh kết quả dự đoán và kết quả thực tế:")
print(comparison_df)

# Đánh giá hiệu suất của mô hình bằng Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred_test)
print("Hiệu suất của mô hình (MSE):", mse)

