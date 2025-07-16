import pandas as pd
import os
from statsmodels.tsa.arima.model import ARIMA
import pickle
import warnings
warnings.filterwarnings('ignore')

# Đọc dữ liệu chi tiêu và phân cụm
spending_data = pd.read_csv('../user_data_ver1/jars_distribution_with_actual.csv')
cluster_data = pd.read_csv('../user_classification/user_clusters.csv')

# Gộp dữ liệu chi tiêu với nhãn cụm
merged_data = pd.merge(spending_data, cluster_data, on='user_id', how='left')

# Áp dụng trọng số cho 12 tháng (ưu tiên 3 tháng gần nhất: 10, 11, 12)
merged_data['weight'] = merged_data['month'].apply(lambda x: 1.0 if x in [10, 11, 12] else 0.5)

# Chuẩn bị chuỗi thời gian có trọng số cho từng hũ hoặc cụm
def prepare_weighted_time_series(data_df):
    weighted_series = data_df.groupby('month').apply(
        lambda x: (x['actual_spent_amount'] * x['weight']).sum() / x['weight'].sum()
    ).sort_index()
    return weighted_series

# Huấn luyện ARIMA cho từng hũ trong từng cụm và lưu mô hình
core_jars = ["NEC", "FFA", "EDU", "LTSS", "PLAY"]
models = {}

for cluster_id in range(4):
    cluster_df = merged_data[merged_data['cluster'] == cluster_id]
    if cluster_df.empty:
        print(f"Không có dữ liệu cho cụm {cluster_id}")
        continue
    
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs("arima_models", exist_ok=True)
    # Huấn luyện cho từng hũ trong cụm
    for jar in core_jars:
        jar_data = cluster_df[cluster_df['jar'] == jar]
        if len(jar_data) == 0:
            print(f"Không có dữ liệu cho hũ {jar} trong cụm {cluster_id}")
            continue
        
        ts_jar = prepare_weighted_time_series(jar_data)
        if len(ts_jar) < 2:
            print(f"Không đủ dữ liệu cho hũ {jar} trong cụm {cluster_id}")
            continue
        
        try:
            model = ARIMA(ts_jar, order=(1, 0, 1))
            model_fit = model.fit()
            models[(cluster_id, jar)] = model_fit
            # Lưu mô hình cho từng hũ
            with open(f"arima_models/cluster_{cluster_id}_{jar.lower()}_arima.pkl", "wb") as f:
                pickle.dump(model_fit, f)
            print(f"Mô hình ARIMA cho hũ {jar} trong cụm {cluster_id} đã được huấn luyện")
        except Exception as e:
            print(f"ARIMA không hội tụ cho hũ {jar} trong cụm {cluster_id}: {e}")