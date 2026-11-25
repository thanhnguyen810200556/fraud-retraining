## Dataset Quality and Exploratory Data Analysis

### 1. Phân bố dữ liệu và tính phù hợp của IEEE-CIS

Bộ dữ liệu IEEE-CIS được phân tích theo 21 khối thời gian liên tiếp, mỗi khối gồm 7 ngày. Tổng cộng, giai đoạn đánh giá chứa 443.845 giao dịch, trong đó có 16.769 giao dịch gian lận, tương ứng tỷ lệ gian lận khoảng 3,78%.

Số lượng giao dịch trong mỗi khối dao động từ 18.507 đến 26.932, với số mẫu gian lận từ 557 đến 1.133. Không có khối nào xuất hiện cảnh báo thiếu mẫu gian lận. Kết quả này cho thấy cách phân chia theo tuần có đủ số lượng mẫu để thực hiện đánh giá hiệu năng mô hình, đặc biệt đối với các chỉ số phù hợp với dữ liệu mất cân bằng như AUCPR và Recall@FPR.

Tỷ lệ gian lận thay đổi từ 3,01% đến 4,93% giữa các khối, cho thấy phân phối nhãn không ổn định theo thời gian. Vì AUCPR chịu ảnh hưởng bởi tỷ lệ lớp dương, sự biến động của chỉ số này cần được phân tích kết hợp với các chỉ số đánh giá khác và kết quả của các chính sách đối chứng trên cùng khối thời gian.

### 2. Sự thay đổi phân phối đặc trưng

Phân tích 381 đặc trưng thuộc cấu hình NoCardID cho thấy sự thay đổi phân phối xuất hiện xuyên suốt giai đoạn đánh giá. Median PSI của các khối dao động từ 0,1703 đến 0,4168, trong khi P90 PSI vượt mức tham khảo 0,25 tại toàn bộ 21 khối.

Trong giai đoạn validation, trung bình khoảng 56,10% đặc trưng mỗi khối được phân loại là có PSI cao. Kết quả này cho thấy IEEE-CIS là một môi trường phù hợp để nghiên cứu các phương pháp phát hiện thay đổi phân phối không cần nhãn.

Tuy nhiên, sự xuất hiện thường xuyên của các tín hiệu drift cũng đặt ra hạn chế đối với những chính sách sử dụng ngưỡng PSI cố định. Nếu mọi tín hiệu drift đều dẫn đến tái huấn luyện, hệ thống có thể thực hiện nhiều lần cập nhật không cần thiết. Do đó, cần đánh giá mối quan hệ giữa tín hiệu drift và lợi ích thực tế của việc tái huấn luyện.

### 3. Sự thay đổi tỷ lệ giá trị thiếu

Kết quả EDA cho thấy missingness drift là một đặc điểm đáng chú ý của IEEE-CIS. Mức thay đổi tỷ lệ missing trung bình trên các đặc trưng dao động từ 14,65 đến 19,36 điểm phần trăm giữa các khối.

Trên tập validation, hệ số tương quan Pearson giữa Mean PSI và Mean Missingness Drift của 381 đặc trưng đạt khoảng 0,8499. Kết quả này gợi ý rằng sự thay đổi tỷ lệ giá trị thiếu có liên quan đáng kể đến tín hiệu drift được đo bằng PSI.

Một số nhóm đặc trưng có PSI cao và mức thay đổi missingness giống nhau, cho thấy các tín hiệu drift có thể tồn tại sự phụ thuộc về cấu trúc. Vì vậy, việc tổng hợp tín hiệu drift cần xem xét khả năng nhiều đặc trưng cùng phản ánh một dạng thay đổi của dữ liệu.

Để bảo toàn thông tin này, nghiên cứu giữ nguyên các giá trị thiếu và sử dụng khả năng xử lý missing values của LightGBM, thay vì áp dụng các phương pháp điền giá trị thiếu hoặc loại bỏ đặc trưng một cách máy móc.

### 4. Đánh giá bộ dữ liệu BAF

Bộ dữ liệu BAF gồm sáu biến thể, mỗi biến thể có một triệu mẫu, bao phủ tám tháng và có tỷ lệ gian lận khoảng 1,103%. Các biến thể có quy mô lớn và phân phối nhãn mất cân bằng, phù hợp với bài toán phát hiện gian lận.

Phân tích BAF Base cho thấy Median PSI tương đối thấp, trong khi P90 PSI tăng từ 0,2003 lên 1,6371 giữa tháng 1 và tháng 7. Kết quả này gợi ý rằng sự thay đổi phân phối tập trung mạnh ở một số đặc trưng.

Tuy nhiên, do BAF chỉ cung cấp thông tin thời gian theo tháng, bộ dữ liệu không phù hợp trực tiếp với giao thức mô phỏng độ trễ nhãn theo ngày của IEEE-CIS. Ngoài ra, Variant III và Variant V có schema khác với Base, cần được kiểm tra trước khi thực hiện so sánh giữa các biến thể.

### 5. Kết luận

Kết quả EDA xác nhận IEEE-CIS đáp ứng các yêu cầu về quy mô mẫu, tính liên tục thời gian, số lượng giao dịch gian lận và sự biến động phân phối cần thiết cho nghiên cứu chính sách tái huấn luyện trong điều kiện nhãn trễ.

Quá trình tiền xử lý được thiết kế theo hướng bảo toàn phân phối dữ liệu gốc, giữ nguyên sự mất cân bằng lớp và thông tin giá trị thiếu, không loại bỏ đặc trưng chỉ vì có mức drift cao. Hai cấu hình FullFeatures và NoCardID được duy trì nhằm đảm bảo khả năng tái lập thực nghiệm và đánh giá ảnh hưởng của các thuộc tính liên quan đến định danh.

IEEE-CIS được lựa chọn làm bộ dữ liệu thực nghiệm chính, trong khi BAF được sử dụng như bộ dữ liệu bổ sung để đánh giá khả năng khái quát hóa ở độ phân giải thời gian theo tháng.

### 6. Preprocessing Decisions

Nhóm thống nhất các quyết định tiền xử lý sau:

- Giữ nguyên dữ liệu raw và lưu dữ liệu chuẩn hóa dưới định dạng Parquet.
- Sử dụng dữ liệu transaction của IEEE-CIS, không ghép bảng identity trong cấu hình thực nghiệm hiện tại (use_identity: false).
- Kiểm tra tính duy nhất của khóa, tính hợp lệ của nhãn và trường thời gian.
- Giữ nguyên missing values và phân phối lớp tự nhiên.
- Không chuẩn hóa thang đo hoặc loại bỏ outliers một cách tự động.
- Không thực hiện feature selection dựa trên mức độ drift.
- Sử dụng NoCardID làm feature set chính và FullFeatures làm feature set đối chứng.
- Tách nhãn, metadata thời gian và thông tin mô phỏng khỏi model features.
- Chỉ học các tham số preprocessing phụ thuộc dữ liệu từ training window hợp lệ.
- Xử lý độc lập IEEE-CIS và từng biến thể BAF.

### 7. Temporal Partitioning Decisions

IEEE-CIS được phân chia thành bốn giai đoạn: Warmup (ngày 1–30), Validation (ngày 31–86), Test (ngày 87–177) và Tail (ngày 178–182).

Validation gồm 8 khối tuần với 169.439 giao dịch và Test gồm 13 khối tuần với 274.406 giao dịch. Tail được giữ lại trong dữ liệu chuẩn hóa nhưng không đưa vào kết quả đánh giá chính.

BAF được phân chia theo tháng, sử dụng tháng 0–1 làm Warmup, tháng 2–3 làm Validation và tháng 4–7 làm Test. Không tạo timestamp ngày giả cho BAF.

### 8. Overall Assessment

Kết quả EDA cho thấy IEEE-CIS đáp ứng các yêu cầu về quy mô, độ phân giải thời gian, số lượng mẫu gian lận và sự thay đổi phân phối cần thiết cho nghiên cứu chính sách tái huấn luyện nhận biết độ trễ nhãn.

BAF phù hợp làm benchmark bổ sung để đánh giá độ bền của phương pháp trên dữ liệu có đặc điểm phân phối và độ phân giải thời gian khác.

Tuy nhiên, kết quả drift không trực tiếp chứng minh sự suy giảm hiệu năng mô hình hoặc lợi ích của retraining. Những vấn đề này cần được kiểm chứng thông qua các thực nghiệm temporal evaluation và so sánh chính sách trong các giai đoạn tiếp theo.
