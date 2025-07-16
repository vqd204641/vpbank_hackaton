import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import os

# Hàm huấn luyện và lưu mô hình
def train_and_save_model(data_path, output_dir="./classify_model", optimal_k=4):
    os.makedirs(output_dir, exist_ok=True)

    # Đọc dữ liệu
    data = pd.read_csv(data_path)
    core_jars = ["NEC", "FFA", "EDU", "LTSS", "PLAY"]
    data_core = data[data['jar'].isin(core_jars)]

    # Tạo bảng pivot: trung bình tỷ trọng qua các tháng
    pivot_percent = data_core.pivot_table(
        values='percent', index='user_id', columns='jar', aggfunc='mean'
    ).fillna(0)
    for jar in core_jars:
        if jar not in pivot_percent.columns:
            pivot_percent[jar] = 0

    # Tính trung bình income
    pivot_income = data_core.groupby('user_id')['income'].mean().reset_index()

    # Phân khoảng income
    bins = np.arange(0, pivot_income['income'].max() + 5_000_000, 5_000_000)
    labels = [f'{int(i/1e6)}-{int((i+5_000_000)/1e6)}M' for i in bins[:-1]]
    pivot_income['income_bin'] = pd.cut(pivot_income['income'], bins=bins, labels=labels, include_lowest=True)
    pivot_income['income_bin_encoded'] = pd.Categorical(pivot_income['income_bin']).codes

    # Kết hợp dữ liệu
    pivot_data = pivot_percent[core_jars].merge(pivot_income[['user_id', 'income_bin_encoded']], on='user_id')
    features = pivot_data[core_jars + ['income_bin_encoded']]

    # Chuẩn hóa và huấn luyện mô hình
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    kmeans.fit(scaled_data)

    # Gán nhãn cụm vào dữ liệu
    pivot_data['cluster'] = kmeans.labels_

    print("\n📊 Số lượng người dùng theo cụm:")
    print(pivot_data['cluster'].value_counts().sort_index())

    cluster_summary = pivot_data.groupby('cluster')[core_jars + ['income_bin_encoded']].mean()
    print("\n📈 Trung bình các đặc trưng trong mỗi cụm:")
    print(cluster_summary)

    # Lưu user_id, cluster và income_bin vào file CSV
    cluster_output_path = "user_clusters.csv"

    # Kết hợp income_bin từ pivot_income để lưu đúng
    full_user_info = pivot_data.merge(
        pivot_income[['user_id', 'income_bin']],
        on='user_id',
        how='left'
    )[['user_id', 'cluster', 'income_bin']]

    full_user_info.to_csv(cluster_output_path, index=False)
    print(f"\n📁 Đã lưu thông tin phân cụm người dùng vào: {cluster_output_path}")

    # Lưu mô hình và scaler
    dump(scaler, f"{output_dir}/scaler.joblib")
    dump(kmeans, f"{output_dir}/kmeans.joblib")
    print(f"\n✅ Mô hình và scaler đã được lưu ở {output_dir}.")

# Hàm phân loại người dùng mới
def classify_new_user(percent_dict=None, income=None, core_jars=["NEC", "FFA", "EDU", "LTSS", "PLAY"]):
    """
    Phân loại người dùng mới dựa trên thu nhập, bỏ qua phân bổ hũ nếu không có percent_dict.
    Income là bắt buộc.

    Args:
        percent_dict (dict, optional): Tỉ lệ phân bổ hũ, ví dụ {"NEC": 55, "FFA": 10, ...}.
        income (float/int): Thu nhập của người dùng (VND), bắt buộc.
        core_jars (list): Danh sách hũ tài chính, mặc định ["NEC", "FFA", "EDU", "LTSS", "PLAY"].

    Returns:
        tuple: (cluster_label, income_bin)
            - cluster_label (int): Nhãn cụm của người dùng.
            - income_bin (str): Khoảng thu nhập (ví dụ: "15-20M").

    Raises:
        ValueError: Nếu income không được cung cấp.
    """
    # Kiểm tra income bắt buộc
    if income is None:
        raise ValueError("Thu nhập (income) là bắt buộc.")

    # Tải scaler và mô hình
    base_path = os.path.join(os.path.dirname(__file__), "classify_model")
    scaler_path = os.path.join(base_path, "scaler.joblib")
    kmeans_path = os.path.join(base_path, "kmeans.joblib")

    try:
        scaler = load(scaler_path)
        kmeans = load(kmeans_path)
    except FileNotFoundError:
        raise FileNotFoundError("Không tìm thấy scaler.joblib hoặc kmeans.joblib trong thư mục classify_model.")

    # Phân khoảng income
    bins = np.arange(0, int(income) + 5_000_000, 5_000_000)
    labels = [f'{int(i/1e6)}-{int((i+5_000_000)/1e6)}M' for i in bins[:-1]]
    try:
        income_bin = pd.cut([income], bins=bins, labels=labels, include_lowest=True)[0]
        income_bin_encoded = pd.Categorical([income_bin]).codes[0]
    except ValueError:
        income_bin = "0-5M"
        income_bin_encoded = 0

    # Nếu không có percent_dict, chỉ sử dụng income_bin_encoded
    if not percent_dict:
        user_features = np.array([[0] * len(core_jars) + [income_bin_encoded]])
    else:
        # Tạo DataFrame từ percent_dict
        user_data = pd.DataFrame([percent_dict])
        user_data = user_data.reindex(columns=core_jars, fill_value=0)
        # Kết hợp với income_bin_encoded
        user_features = np.column_stack((user_data[core_jars].values, [income_bin_encoded]))

    # Chuẩn hóa và dự đoán
    try:
        user_features_scaled = scaler.transform(user_features)
        cluster_label = kmeans.predict(user_features_scaled)[0]
    except Exception as e:
        raise ValueError(f"Lỗi khi dự đoán cụm: {str(e)}")
    
    return cluster_label, income_bin

# Chạy huấn luyện nếu script được gọi trực tiếp
# if __name__ == "__main__":
    #train_and_save_model("../user_data_ver1/jars_distribution_with_actual.csv")
    #classify_new_user(income= 14000000)