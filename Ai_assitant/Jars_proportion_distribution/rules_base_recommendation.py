import sys
import os

# Lấy đường dẫn thư mục gốc 'vpbank_hackathon'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

# Giờ có thể import theo đúng đường dẫn thư mục
from spending_forecast.forecast import ARIMAPredictor
from user_classification.classification import classify_new_user

# Khởi tạo và dự đoán
# predictor = ARIMAPredictor()
# result = predictor.predict_cluster_spending(0)
# for jar, value in result.items():
#     print(f"{jar}: {value:.2f} VND")

cluster, income_bin = classify_new_user(income=14000000)
print (cluster, income_bin)