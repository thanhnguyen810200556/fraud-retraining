# Fraud Retraining

**When to Retrain? Label-Delay-Aware Retraining Policies for Fraud Detection on Evolving Data Streams**

Đồ án môn Big Data — nghiên cứu chính sách tái huấn luyện nhận biết độ trễ nhãn cho phát hiện gian lận trên dòng dữ liệu thay đổi theo thời gian.

## 1. Tổng quan

Đề tài nghiên cứu khi nào nên tái huấn luyện mô hình phát hiện gian lận trong điều kiện phân phối dữ liệu thay đổi và nhãn giao dịch không có ngay lập tức. Mục tiêu là cân bằng hiệu năng dự đoán với số lần và chi phí tái huấn luyện.

Hướng thực nghiệm: LightGBM, các chính sách NoRetrain, Periodic, Random, Oracle (đối chứng tham khảo) và Adaptive/Trigger; tín hiệu drift PSI, KS và Domain Classifier. Các chỉ số dự kiến gồm AUCPR, ROC-AUC, Recall@FPR=1%, số lần retrain và chi phí tính toán.

**Trạng thái:** repository hiện tập trung vào Dataset Collection & Preparation (Bước 1–5); Bước 6 cần tích hợp và chạy kiểm định trên dữ liệu cục bộ. Pipeline mô hình và so sánh chính sách hoàn chỉnh là giai đoạn tiếp theo; chưa có lệnh chạy end-to-end được xác nhận.

## 2. Cấu trúc repository

```text
fraud-retraining/
├── README.md
├── DATASET.md                 # Tài liệu dữ liệu riêng (cần bổ sung)
├── requirements.txt
├── configs/                   # Cấu hình dataset và features
├── data/
│   ├── raw/                   # CSV gốc và manifest
│   ├── processed/             # Parquet chuẩn hóa
│   ├── temporal/              # Temporal indexes
│   └── simulations/
│       └── label_schedules/    # Lịch khả dụng nhãn
├── notebooks/                 # EDA và pipeline chuẩn bị dữ liệu
├── src/                       # Logic xử lý và mô phỏng
├── tests/                     # Unit tests
└── reports/                   # Thống kê, metadata, hình ảnh, validation
```

Các thư mục dữ liệu sinh ra trong quá trình chạy có thể chưa tồn tại sau khi clone.

## 3. Cài đặt môi trường

Cần Git, Python và pip. Khuyến nghị Python 3.11 (chưa xác nhận phiên bản tối thiểu của toàn bộ project).

```bash
git clone https://github.com/thanhnguyen810200556/fraud-retraining.git
cd fraud-retraining
python -m venv .venv
```

Kích hoạt môi trường ảo trên Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Trên macOS/Linux:

```bash
source .venv/bin/activate
```

Cài thư viện và kernel:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name fraud-retraining --display-name "Python (fraud-retraining)"
```

Nếu repo private, cần được cấp quyền truy cập trước khi clone. Luôn chạy lệnh từ thư mục gốc repository.

## 4. Chuẩn bị dữ liệu

Tải IEEE-CIS từ https://www.kaggle.com/competitions/ieee-fraud-detection/data và BAF từ https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022 . Đăng nhập và chấp nhận điều khoản nếu được yêu cầu.

Giải nén và đặt đúng tên file:

```text
data/raw/
├── dataset_manifest.csv
├── ieee_cis/
│   ├── train_transaction.csv
│   └── train_identity.csv
└── baf/
    ├── Base.csv
    ├── Variant I.csv
    ├── Variant II.csv
    ├── Variant III.csv
    ├── Variant IV.csv
    └── Variant V.csv
```

IEEE-CIS là dataset chính (590.540 giao dịch); BAF là dataset bổ sung (6 biến thể, mỗi biến thể 1 triệu mẫu). Cấu hình hiện tại không ghép identity vào transaction, nhưng vẫn giữ file identity phục vụ profiling. Không commit dữ liệu gốc hoặc Parquet lớn lên Git. Kiểm tra quyền sử dụng và phiên bản nguồn, đặc biệt BAF, trước khi phân phối dữ liệu.

## 5. Chạy pipeline dữ liệu

Mở Jupyter từ thư mục gốc:

```bash
python -m jupyter notebook
```

Chọn kernel `Python (fraud-retraining)` và chạy Run All từng notebook theo thứ tự sau (tên được ghi theo stem, mở file `.ipynb` tương ứng):

| Thứ tự | Notebook                    | Mục đích                                         |
| ------ | --------------------------- | ------------------------------------------------ |
| 1      | `01_data_profiling`         | Kiểm tra dữ liệu gốc, schema, chất lượng         |
| 2      | `02_temporal_eda`           | Phân tích dữ liệu theo thời gian                 |
| 3      | `03_drift_exploration`      | Khảo sát distribution drift                      |
| 4      | `04_baf_eda`                | Khảo sát BAF                                     |
| 5      | `05_preprocessing`          | Chuẩn hóa dữ liệu, tạo Parquet và feature schema |
| 6      | `06_temporal_partitioning`  | Tạo temporal indexes và block manifests          |
| 7      | `07_label_delay_simulation` | Tạo label schedules và thống kê dữ liệu có nhãn  |
| 8      | `08_baseline_policies`      | Baseline NoRetrain, Periodic, Random-k (T11–T16) |

Số thứ tự notebook không đồng nhất với sáu bước nghiệp vụ Dataset Preparation. Không bỏ qua notebook nếu chưa có các artifact đầu vào tương ứng. Nếu notebook không tìm thấy module `src`, kiểm tra working directory và kernel; không sửa đường dẫn dữ liệu tùy tiện. Chạy:

```bash
python -m pytest -q
python -m src.validation
```

Chỉ coi dataset đã freeze khi lần chạy thành công và `reports/validation/freeze_manifest.json` được tạo cho đúng phiên bản artifact hiện tại. Commit mã nguồn trước khi freeze để ghi nhận đúng phiên bản. Nếu validation thất bại, xem `reports/validation/final_validation.csv`, sửa nguyên nhân và chạy lại giai đoạn liên quan; không chỉnh checksum để bỏ qua lỗi.

## 6. Cấu hình thực nghiệm hiện tại

| Tham số                     | IEEE-CIS          | BAF           |
| --------------------------- | ----------------- | ------------- |
| Warmup                      | Ngày 1–30         | Tháng 0–1     |
| Validation                  | Ngày 31–86        | Tháng 2–3     |
| Test                        | Ngày 87–177       | Tháng 4–7     |
| Block                       | 7 ngày            | 1 tháng       |
| Label delay                 | 0, 7, 14, 30 ngày | 0, 1, 2 tháng |
| Training window             | 30, 60, 90 ngày   | 2, 3 tháng    |
| Cấu hình chính delay/window | 7/60 ngày         | 1/3 tháng     |

IEEE-CIS giữ Tail ngày 178–182 ngoài đánh giá chính. Feature set chính là NoCardID (381 features), FullFeatures (391 features) dùng đối chứng. Seed mặc định 42. Xem `configs/dataset.yaml` và `configs/features.yaml` để biết cấu hình đầy đủ.

Không dùng nhãn chưa khả dụng tại thời điểm quyết định, không dùng test để chọn tham số, không đưa ID, nhãn và metadata mô phỏng vào model features. Mọi chính sách phải sử dụng cùng phiên bản dataset đã freeze.

## 7. Kiểm tra kết quả

| Đầu ra              | Vị trí                                      |
| ------------------- | ------------------------------------------- |
| Dữ liệu chuẩn hóa   | `data/processed/`                           |
| Feature schema      | `reports/preprocessing/feature_schema.json` |
| Temporal indexes    | `data/temporal/`                            |
| Temporal reports    | `reports/temporal/`                         |
| Label schedules     | `data/simulations/label_schedules/`         |
| Label delay reports | `reports/label_delay/`                      |
| Báo cáo chất lượng  | `reports/dataset_quality_report.md`         |
| Final validation    | `reports/validation/`                       |

Nếu đã nhận bộ artifact được freeze từ thành viên khác theo điều khoản dữ liệu, kiểm tra checksum và phiên bản trước khi sử dụng; không cần tái tạo toàn bộ nếu chúng khớp.
