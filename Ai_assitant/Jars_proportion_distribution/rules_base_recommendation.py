import sys
import os
import pandas as pd
# Lấy đường dẫn thư mục gốc 'vpbank_hackathon'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)
# Giờ có thể import theo đúng đường dẫn thư mục
from user_classification.classification import classify_new_user
import google.generativeai as genai
from dotenv import load_dotenv

clusters_df = pd.read_csv("../../user_classification/user_clusters.csv")
jars_df = pd.read_csv("../../user_data_ver1/jars_distribution_with_actual.csv")

load_dotenv(dotenv_path="../../.env")

# Lấy API key từ biến môi trường
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Tính tỷ lệ hoàn thành như trước
def compute_completion_rates(cluster):
    cluster_users = clusters_df[clusters_df['cluster'] == cluster]['user_id']
    filtered_jars = jars_df[jars_df['user_id'].isin(cluster_users)].copy()
    filtered_jars['is_completed'] = filtered_jars['spending_percent'] < 100

    result = (
        filtered_jars
        .groupby(['jar', 'percent'])
        .agg(
            total_entries=('is_completed', 'count'),
            completed_count=('is_completed', 'sum')
        )
        .reset_index()
    )
    result['completion_rate'] = (result['completed_count'] / result['total_entries']) * 100
    return result

def suggest_percents_for_user(income, user_jars,percent_dict=None):
    six_jars_standard = {
    "NEC": 55,
    "FFA": 10,
    "EDU": 10,
    "LTSS": 10,
    "PLAY": 10,
    "GIVE": 5,
    }
    # Lấy cluster của người dùng
    user_cluster, income_bin = classify_new_user(income=income)

    # 2. Tính tỉ lệ hoàn thành trong cluster
    cluster_result = compute_completion_rates(user_cluster)

    # 3. Với mỗi jar, lấy percent tốt nhất
    best_percent_per_jar = (
        cluster_result[cluster_result['jar'].isin(user_jars)]
        .loc[lambda df: df.groupby('jar')['completion_rate'].idxmax()]
        .set_index('jar')
    )

    # 4. Tạo gợi ý phần trăm ban đầu
    suggestions = {}
    for jar in user_jars:
        if jar in best_percent_per_jar.index:
            suggestions[jar] = best_percent_per_jar.loc[jar, 'percent']
        elif jar in six_jars_standard:
            suggestions[jar] = six_jars_standard[jar]
        else:
            suggestions[jar] = 0  # fallback nếu không có gì

    if percent_dict:
        for jar in suggestions:
            if jar in percent_dict:
                old = percent_dict[jar]
                new = suggestions[jar]
                max_diff = 0.1 * old  # 10% ngưỡng cho phép

                # Nếu lệch quá ngưỡng → giới hạn lại
                if abs(new - old) > max_diff:
                    if new > old:
                        suggestions[jar] = round(old + max_diff, 2)
                    else:
                        suggestions[jar] = round(old - max_diff, 2)

    # 5. Chuẩn hóa lại tỉ lệ sao cho tổng = 100%
    total_percent = sum(suggestions.values())
    if total_percent > 100:
        # Scale down
        scale = 100 / total_percent
        suggestions = {jar: round(percent * scale, 2) for jar, percent in suggestions.items()}
    else:
        # Scale up cho đủ 100%
        scale = 100 / total_percent if total_percent > 0 else 0
        suggestions = {jar: round(percent * scale, 2) for jar, percent in suggestions.items()}
    jar_labels = {
        "NEC": "Nhu cầu thiết yếu",                        # Necessities
        "FFA": "Tự do tài chính",                          # Financial Freedom Account
        "EDU": "Giáo dục & phát triển bản thân",           # Education
        "LTSS": "Tiết kiệm dài hạn cho chi tiêu lớn",      # Long-Term Savings for Spending
        "PLAY": "Hưởng thụ",                               # Play
        "GIVE": "Chia sẻ & từ thiện",                      # Giving
        "HEALTH": "Sức khỏe",                              # Health
        "FAMILY": "Gia đình",                              # Family
        "EMERGENCY": "Quỹ khẩn cấp",                       # Emergency Fund
        "DREAM": "Ước mơ & mục tiêu lớn"                   # Dream Jar
    }

    jar_text = "\n".join([
        f"- {jar}: {jar_labels.get(jar, jar)} – {float(suggestions[jar]):.2f}%"
        for jar in user_jars if jar in suggestions
    ])    
    # 6. Trả về dưới dạng DataFrame
    prompt = f"""
    Tôi muốn bạn đóng vai một cố vấn tài chính cá nhân.  
    Dưới đây là tỷ lệ phân bố chi tiêu đề xuất cho một người dùng dựa trên:
    - Dữ liệu thực tế của những người có thu nhập tương tự và có tỷ lệ hoàn thành mục tiêu tài chính cao  
    - Kết hợp với lời khuyên của chuyên gia tài chính

    Tỷ lệ phân bổ hũ như sau:
    {jar_text}

    Hãy đưa ra một **lời khuyên ngắn gọn (4-6 câu) và dễ hiểu** cho người dùng về cách phân bố thu nhập theo các hũ trên.  
    Vui lòng trình bày lời khuyên theo phong cách **giao tiếp tự nhiên**, **động viên**, và giúp người dùng cảm thấy yên tâm khi áp dụng.
    Tóm tắt lại % các hũ đã chia trong 1 dòng ở cuối câu.
    """
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    response = model.generate_content(prompt)

    return response.text

# suggestions = suggest_percents_for_user(income= 15000000, user_jars = ["NEC", "EDU", "FFA", "PLAY","LTSS"])
# print(suggestions)