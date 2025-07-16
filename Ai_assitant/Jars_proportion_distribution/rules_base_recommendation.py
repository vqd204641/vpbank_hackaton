import sys
import os
import pandas as pd
# Lấy đường dẫn thư mục gốc 'vpbank_hackathon'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)
# Giờ có thể import theo đúng đường dẫn thư mục
from user_classification.classification import classify_new_user

clusters_df = pd.read_csv("../../user_classification/user_clusters.csv")
jars_df = pd.read_csv("../../user_data_ver1/jars_distribution_with_actual.csv")

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

    # 6. Trả về dưới dạng DataFrame
    return pd.DataFrame([
        {'jar': jar, 'suggested_percent': percent}
        for jar, percent in suggestions.items()
    ]).sort_values(by='suggested_percent', ascending=False).reset_index(drop=True)

# suggestions = suggest_percents_for_user(income= 15000000, user_jars = ["NEC", "EDU", "FFA", "PLAY","LTSS"])
# print(suggestions)