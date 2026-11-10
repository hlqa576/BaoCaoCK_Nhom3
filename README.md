# DỰ ÁN PHÂN TÍCH DỮ LIỆU VÀ DỰ ĐOÁN GIÁ BẤT ĐỘNG SẢN

## 1. Giới thiệu dự án

Dự án được thực hiện nhằm phân tích dữ liệu bất động sản, khám phá các yếu tố ảnh hưởng đến giá nhà và ứng dụng các thuật toán Machine Learning để phân tích, dự đoán và phân nhóm dữ liệu.

## 2. Mục tiêu dự án

- Thu thập và xử lý dữ liệu bất động sản.
- Phân tích và trực quan hóa dữ liệu.
- Khám phá mối quan hệ giữa các thuộc tính và giá bất động sản.
- Xây dựng và đánh giá các mô hình Machine Learning.
- Ứng dụng kết quả mô hình vào bài toán dự đoán giá nhà.

## 3. Công nghệ sử dụng

- Python
- Jupyter Notebook
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Ipywidgets

## 4. Các nội dung thực hiện

- Tiền xử lý và làm sạch dữ liệu.
- Trực quan hóa dữ liệu bằng biểu đồ.
- Xây dựng mô hình Linear Regression.
- Xây dựng mô hình Random Forest.
- Áp dụng Logistic Regression.
- Phân cụm dữ liệu bằng K-Means.
- Xây dựng Decision Tree và Grid Search.
- Đánh giá mô hình bằng K-Fold Cross-Validation.

## 5. Cấu trúc dự án

## 5. Cấu trúc dự án

```text
BaoCaoCK_Nhom3/
│
├── app.py
├── BaoCaoCK_nhom3_Dudongianha.ipynb
├── README.md
├── requirements.txt
├── .gitignore
└── modified_data.csv
```

## 6. Hướng dẫn cài đặt và sử dụng

**Bước 1:** Clone dự án về máy tính.

```bash
git clone <LINK_GITHUB_CUA_DU_AN>
```

**Bước 2:** Di chuyển vào thư mục dự án.

```bash
cd BaoCaoCK_Nhom3
```

**Bước 3:** Cài đặt các thư viện cần thiết.

```bash
pip install -r requirements.txt
```

**Bước 4:** Khởi chạy Jupyter Notebook.

```bash
jupyter notebook
```

Mở file `BaoCaoCK_nhom3_Dudongianha.ipynb` để xem và thực thi nội dung phân tích.

## 7. Thành viên thực hiện
```text
| STT |     Họ và tên     |    MSSV    |
| 1   | [Đỗ Quốc Duy]     | [24690901] |
| 2   | [Huỳnh Lê Quốc An]| [24704711] |
| 3   | [Lê Phước Thịnh]  | [24651371] |
```

## 8. Kết luận

Dự án giúp nhóm vận dụng kiến thức phân tích dữ liệu và Machine Learning vào bài toán bất động sản thực tế. Thông qua quá trình xử lý dữ liệu, trực quan hóa và xây dựng mô hình, nhóm có cơ hội tìm hiểu và đánh giá khả năng ứng dụng các thuật toán học máy trong việc phân tích và dự đoán giá nhà.

## Ứng dụng dự đoán giá nhà trực tuyến

Nhóm đã triển khai ứng dụng dự đoán giá nhà bằng Streamlit.

**Truy cập ứng dụng tại đây:**

[🏠 Chatbot dự đoán giá nhà](https://du-doan-gia-nha-nhom3.streamlit.app/)

Người dùng có thể nhập các thông tin về căn nhà để hệ thống dự đoán giá bán bằng mô hình Machine Learning.
