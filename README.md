# CS2225.CH1801.N014

by [Bùi Tổng Nha - CH1801033](https://github.com/nhabt-uit/CS2225.CH1801.N014/tree/NhaBT) và [Châu Ngọc Long Giang - CH1801026](https://github.com/nhabt-uit/CS2225.CH1801.N014/tree/Giangcnl)


1. Dataset<br>
Bộ dataset được lấy từ German Traffic Sign Benchmark. Link: http://kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign<br>

2. Import các thư viện cần thiết cho chương trình<br>
Đầu tiên nhập các thư viện cần thiết như keras để xây dựng mô hình chính, sklearn để tách dữ liệu đào tạo và kiểm tra, PIL để chuyển đổi hình ảnh thành mảng số và các thư viện khác như pandas, numpy , matplotlib và tensorflow.
```sh
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
import os
os.chdir('/content/drive/My Drive/VRA - Self Driving Cars')
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
```

3. Truy xuất hình ảnh<br>
Bước này sẽ truy xuất hình ảnh và nhãn. Sau đó thay đổi kích thước hình ảnh thành (30,30) vì tất cả các hình ảnh phải có cùng kích thước để nhận dạng. Sau đó chuyển đổi hình ảnh thành mảng numpy.<br>

```sh
data = []
labels = []
classes = 5
cur_path = os.getcwd()

for i in range(classes):
    path = os.path.join(cur_path,'/content/drive/My Drive/VRA - Self Driving Cars/vra/train',str(i))
    images = os.listdir(path)
    for a in images:
        try:
            image = Image.open(path + '/'+ a)
            image = image.resize((30,30))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except Exception as e:
            print(e)
            
data = np.array(data)
labels = np.array(labels)
```

4. Tách tập dữ liệu<br>
Chia tập data thành bộ train và bộ test. 80% data là train và 20% data để test.<br>

```sh
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
```

5. Xây dựng mô hình<br>
Sử dụng mô hình tuần tự từ thư viện keras. Tiếp theo là thêm các layer để tạo convolutional neural network. Trong 2 layer Conv2D đầu tiên, sử dụng 32 bộ lọc và kích thước kernel là (5,5).<br> Trong layer MaxPool2D, kích thước nhóm (2,2) có nghĩa là nó sẽ chọn giá trị tối đa của mỗi khu vực 2 x 2 của hình ảnh. Bằng cách thực hiện kích thước này của ảnh sẽ giảm theo hệ số 2. Trong dropout layer, nhóm đã giữ tỷ lệ dropout layer = 0,25 có nghĩa là 25% neurons được loại bỏ ngẫu nhiên.<br> Dùng layer phẳng để chuyển đổi dữ liệu 2-D sang vectơ 1-D. Layer này được theo sau bởi dense layer, dropout layer và dense layer một lần nữa. Dense layer cuối cùng xuất ra 6 nút khi các biển báo giao thông được chia thành 6 loại trong tập dataset. Layer này sử dụng hàm kích hoạt softmax cho giá trị xác suất và dự đoán tùy chọn nào trong số 6 tùy chọn có xác suất cao nhất.<br>

```sh
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))
```

7. Độ chính xác trên bộ thử nghiệm<br>
Có độ chính xác 94,91% trên bộ thử nghiệm.
