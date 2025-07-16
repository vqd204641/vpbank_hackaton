import os
import pickle
import warnings
warnings.filterwarnings('ignore')

class ARIMAPredictor:
    def __init__(self, model_dir=None):
        """
        Khởi tạo class với thư mục chứa mô hình.
        
        Args:
            model_dir (str): Đường dẫn đến thư mục chứa file .pkl (mặc định là "arima_models")
        """
        if model_dir is None:
            # Tự động lấy đường dẫn thư mục chứa forecast.py rồi đi đến arima_models
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.model_dir = os.path.join(current_dir, "arima_models")
        else:
            self.model_dir = model_dir

        self.models = {}
        self.core_jars = ["NEC", "FFA", "EDU", "LTSS", "PLAY"]
        self._load_models()

    def _load_models(self):
        """Tải tất cả mô hình từ file .pkl trong thư mục model_dir."""
        for cluster_id in range(4):  # Cụm 0, 1, 2, 3
            # # Tải mô hình tổng quát cho cụm
            # model_path = os.path.join(self.model_dir, f"cluster_{cluster_id}_arima.pkl")
            # if os.path.exists(model_path):
            #     with open(model_path, 'rb') as f:
            #         self.models[cluster_id] = pickle.load(f)
            #     print(f"Đã tải mô hình cho cụm {cluster_id}")
            # else:
            #     print(f"Không tìm thấy file {model_path}")

            # Tải mô hình cho từng hũ trong cụm
            for jar in self.core_jars:
                model_path = os.path.join(self.model_dir, f"cluster_{cluster_id}_{jar.lower()}_arima.pkl")
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.models[(cluster_id, jar)] = pickle.load(f)
                    print(f"Đã tải mô hình cho hũ {jar} trong cụm {cluster_id}")
                else:
                    print(f"Không tìm thấy file {model_path}")

    def predict_next_month_spending(self, cluster_id, jar=None, month=13):
        """
        Dự đoán chi tiêu cho tháng tiếp theo.
        
        Args:
            cluster_id (int): ID của cụm (0-3)
            jar (str, optional): Tên hũ (NEC, FFA, EDU, LTSS, PLAY), nếu None thì dự đoán tổng
            month (int): Tháng dự đoán (mặc định là 13)
            
        Returns:
            float: Giá trị chi tiêu dự đoán làm tròn đến 100k, hoặc None nếu không có mô hình
        """
        key = (cluster_id, jar) if jar else cluster_id
        if key not in self.models:
            print(f"Không có mô hình cho cụm {cluster_id}" + (f" hoặc hũ {jar}" if jar else ""))
            return None
        
        model = self.models[key]
        forecast = model.forecast(steps=1).iloc[0]  # Sử dụng iloc[0] để lấy giá trị
        # Làm tròn đến 100,000
        rounded_forecast = round(forecast / 100000) * 100000
        return rounded_forecast

    def predict_cluster_spending(self, cluster_id):
        """
        Dự đoán chi tiêu cho tất cả hũ và tổng chi tiêu trong cụm.
        
        Args:
            cluster_id (int): ID của cụm (0-3)
            
        Returns:
            dict: Từ điển chứa dự đoán cho từng hũ và tổng chi tiêu
        """
        predictions = {}
        total = 0

        # Dự đoán cho từng hũ
        for jar in self.core_jars:
            prediction = self.predict_next_month_spending(cluster_id, jar)
            if prediction is not None:
                predictions[jar] = prediction
                total += prediction

        # Thêm tổng chi tiêu
        predictions['total'] = total

        return predictions

# Ví dụ sử dụng (có thể xóa khi nhập từ folder khác)
if __name__ == "__main__":
    # Khởi tạo class
    predictor = ARIMAPredictor()

    # Dự đoán cho cụm 0
    result = predictor.predict_cluster_spending(0)
    print(f"\nDự đoán chi tiêu cho cụm 0 (làm tròn đến 100k):")
    for jar, value in result.items():
        print(f"{jar}: {value:.2f} VND")