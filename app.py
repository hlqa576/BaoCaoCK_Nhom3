import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# =========================================================
# CẤU HÌNH TRANG
# =========================================================
st.set_page_config(
    page_title="Chatbot dự đoán giá bất động sản",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 CHATBOT DỰ ĐOÁN GIÁ BẤT ĐỘNG SẢN AI")
st.caption("Ứng dụng Machine Learning – Decision Tree Regressor")
st.write(
    "Nhập thông tin căn nhà để hệ thống ước tính giá bán "
    "dựa trên dữ liệu bất động sản của nhóm."
)

# =========================================================
# TÌM FILE CSV: hỗ trợ cả data/modified_data.csv và
# modified_data.csv nằm cùng cấp với app.py
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
possible_paths = [
    BASE_DIR / "data" / "modified_data.csv",
    BASE_DIR / "modified_data.csv",
]

DATA_PATH = next((p for p in possible_paths if p.exists()), None)

if DATA_PATH is None:
    st.error(
        "Không tìm thấy modified_data.csv. "
        "Hãy đặt file trong thư mục data/ hoặc cùng cấp với app.py."
    )
    st.stop()


# =========================================================
# ĐỌC, LÀM SẠCH VÀ HUẤN LUYỆN MÔ HÌNH
# Giữ cách xử lý dữ liệu theo notebook:
# - price > 0, bedrooms > 0, bathrooms > 0
# - Lọc giá từ 100.000 đến 3.000.000 USD
# - house_age = 2026 - yr_built
# - log_price = log1p(price)
# - 5 biến đầu vào của chatbot
# - DecisionTreeRegressor + GridSearchCV (5-fold)
# =========================================================
@st.cache_resource
def train_model(csv_path):
    data = pd.read_csv(csv_path)

    required_columns = [
        "price", "bedrooms", "bathrooms",
        "sqft_living", "waterfront", "yr_built"
    ]
    missing = [col for col in required_columns if col not in data.columns]
    if missing:
        raise ValueError(
            "CSV thiếu các cột cần thiết: " + ", ".join(missing)
        )

    clean = data[
        (data["price"] > 0)
        & (data["bedrooms"] > 0)
        & (data["bathrooms"] > 0)
    ].copy()

    # Đồng nhất cách tạo biến với notebook của nhóm
    clean["house_age"] = 2026 - clean["yr_built"]

    # Lọc ngoại lệ giá theo notebook
    final_data = clean[
        (clean["price"] >= 100000)
        & (clean["price"] <= 3000000)
    ].copy()

    final_data["log_price"] = np.log1p(final_data["price"])

    features = [
        "sqft_living",
        "bedrooms",
        "bathrooms",
        "house_age",
        "waterfront",
    ]

    X = final_data[features]
    y = final_data["log_price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    tree = DecisionTreeRegressor(random_state=42)

    param_grid = {
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
    }

    grid = GridSearchCV(
        estimator=tree,
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    predictions = best_model.predict(X_test)

    metrics = {
        "r2": r2_score(y_test, predictions),
        "mae_log": mean_absolute_error(y_test, predictions),
        "rmse_log": np.sqrt(mean_squared_error(y_test, predictions)),
        "best_params": grid.best_params_,
        "rows": len(final_data),
        "cv_r2": grid.best_score_,
    }

    return best_model, metrics


with st.spinner("Đang đọc dữ liệu và huấn luyện mô hình Decision Tree lần đầu..."):
    try:
        model, metrics = train_model(str(DATA_PATH))
    except Exception as error:
        st.error(f"Không thể huấn luyện mô hình: {error}")
        st.stop()

st.success("Mô hình đã sẵn sàng!")

with st.expander("Thông tin mô hình"):
    st.write(f"Số dòng dữ liệu sau khi làm sạch và lọc giá: **{metrics['rows']:,}**")
    st.write(f"R² trên tập kiểm thử (thang log): **{metrics['r2']:.4f}**")
    st.write(f"R² Cross-Validation tốt nhất: **{metrics['cv_r2']:.4f}**")
    st.write("Siêu tham số tốt nhất:")
    st.json(metrics["best_params"])
    st.caption(
        "Các chỉ số trên được tính trên thang log_price, "
        "không phải sai số USD trực tiếp."
    )

st.divider()
st.subheader("Nhập thông tin căn nhà")

col1, col2 = st.columns(2)

with col1:
    sqft = st.number_input(
        "Diện tích nhà (sqft)",
        min_value=100,
        max_value=20000,
        value=2000,
        step=50,
    )
    bedrooms = st.number_input(
        "Số phòng ngủ",
        min_value=1,
        max_value=20,
        value=3,
        step=1,
    )
    bathrooms = st.number_input(
        "Số phòng tắm",
        min_value=0.5,
        max_value=20.0,
        value=2.0,
        step=0.5,
    )

with col2:
    house_age = st.number_input(
        "Tuổi nhà (năm)",
        min_value=0,
        max_value=300,
        value=15,
        step=1,
    )
    waterfront_label = st.selectbox(
        "Nhà có vị trí ven sông/biển không?",
        ["Không", "Có"],
    )
    waterfront = 1 if waterfront_label == "Có" else 0

if st.button("🔎 DỰ ĐOÁN GIÁ NHÀ", type="primary", use_container_width=True):
    input_data = pd.DataFrame(
        [[sqft, bedrooms, bathrooms, house_age, waterfront]],
        columns=[
            "sqft_living",
            "bedrooms",
            "bathrooms",
            "house_age",
            "waterfront",
        ],
    )

    log_prediction = model.predict(input_data)[0]
    predicted_price = float(np.expm1(log_prediction))

    if predicted_price > 800000:
        segment = "Phân khúc Cao cấp (Luxury)"
    elif predicted_price > 400000:
        segment = "Phân khúc Tầm trung (Mid-tier)"
    else:
        segment = "Phân khúc Phổ thông (Standard)"

    st.divider()
    st.subheader("🤖 Chatbot AI phản hồi")
    st.metric("Giá nhà dự báo ước tính", f"${predicted_price:,.0f} USD")
    st.info(f"Phân khúc thị trường: **{segment}**")

    st.write("**Thông tin căn nhà đã nhập:**")
    st.write(
        f"- Diện tích: {sqft:,} sqft\n"
        f"- Phòng ngủ: {bedrooms}\n"
        f"- Phòng tắm: {bathrooms}\n"
        f"- Tuổi nhà: {house_age} năm\n"
        f"- Vị trí ven sông/biển: {waterfront_label}"
    )

    st.caption(
        "Kết quả là giá ước tính từ mô hình học trên dữ liệu lịch sử, "
        "không phải báo giá hoặc định giá chính thức."
    )

st.divider()
st.caption("Đồ án Phân tích dữ liệu & AI – Nhóm 3")
