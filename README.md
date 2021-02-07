# CS2225.CH1801.N014

by Bùi Tổng Nha - CH1801033 và Châu Ngọc Long Giang - CH1801026

1. Dataset<br>
Bộ dataset được lấy từ German Traffic Sign Benchmark. Link:http://kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign<br>
Import các thư viện cần thiết cho chương trình<br>
Đầu tiên nhập các thư viện cần thiết như keras để xây dựng mô hình chính, sklearn để tách dữ liệu đào tạo và kiểm tra, PIL để chuyển đổi hình ảnh thành mảng số và các thư viện khác như pandas, numpy , matplotlib và tensorflow.


2. Truy xuất hình ảnh<br>
Bước này sẽ truy xuất hình ảnh và nhãn. Sau đó thay đổi kích thước hình ảnh thành (30,30) vì tất cả các hình ảnh phải có cùng kích thước để nhận dạng. Sau đó chuyển đổi hình ảnh thành mảng numpy.<br>


3. Tách tập dữ liệu<br>
Chia tập data thành bộ train và bộ test. 80% data là train và 20% data để test.<br>

4. Xây dựng mô hình<br>
Sử dụng mô hình tuần tự từ thư viện keras. Tiếp theo là thêm các layer để tạo convolutional neural network. Trong 2 layer Conv2D đầu tiên, sử dụng 32 bộ lọc và kích thước kernel là (5,5).<br>

Trong layer MaxPool2D, kích thước nhóm (2,2) có nghĩa là nó sẽ chọn giá trị tối đa của mỗi khu vực 2 x 2 của hình ảnh. Bằng cách thực hiện kích thước này của ảnh sẽ giảm theo hệ số 2. Trong dropout layer, nhóm đã giữ tỷ lệ dropout layer = 0,25 có nghĩa là 25% neurons được loại bỏ ngẫu nhiên.<br>

Dùng layer phẳng để chuyển đổi dữ liệu 2-D sang vectơ 1-D. Layer này được theo sau bởi dense layer, dropout layer và dense layer một lần nữa. Dense layer cuối cùng xuất ra 6 nút khi các biển báo giao thông được chia thành 6 loại trong tập dataset. Layer này sử dụng hàm kích hoạt softmax cho giá trị xác suất và dự đoán tùy chọn nào trong số 6 tùy chọn có xác suất cao nhất.<br>

5. Áp dụng mô hình và vẽ đồ thị cho độ chính xác và mất mát<br>
Chúng tôi sẽ biên dịch mô hình và áp dụng nó bằng chức năng phù hợp. Kích thước lô sẽ là 32. Sau đó, chúng tôi sẽ vẽ biểu đồ cho độ chính xác và thua lỗ. Chúng tôi có độ chính xác xác thực trung bình là 97,6% và độ chính xác đào tạo trung bình là 93,3%.

6. Độ chính xác trên bộ thử nghiệm<br>
Chúng tôi có độ chính xác 94,7% trên bộ thử nghiệm.
