import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import sys
import os
project_root = os.path.abspath(os.path.join(os.getcwd(), "../../../"))

sys.path.append(project_root)
from spending_forecast.forecast import ARIMAPredictor
predictor = ARIMAPredictor() 
predict_cluster = {}

for i in range(4):
    result = predictor.predict_cluster_spending(i)
    spending = result['EDU'] + result['PLAY']
    saving = result['FFA'] + result['LTSS']
    predict_cluster[i] = {"spending": spending, "saving": saving}


# Đọc file CSV
df_goals = pd.read_csv("../../../user_data_ver1/user_goals_2saving_1spending.csv")
df_clusters = pd.read_csv("../../../user_classification/user_clusters.csv")  

# Chuyển đổi ngày tháng
df_goals['start_date'] = pd.to_datetime(df_goals['start_date'])
df_goals['target_date'] = pd.to_datetime(df_goals['target_date'])
df_goals['duration_days'] = (df_goals['target_date'] - df_goals['start_date']).dt.days

# Gộp cluster vào df_goals
df_goals = df_goals.merge(df_clusters[['user_id', 'cluster']], on='user_id', how='left')
df_goals['cluster'] = df_goals['cluster'].fillna(0)  # Mặc định cluster=0 nếu không tìm thấy

# Hàm tính toán các đặc trưng của mục tiêu
def extract_features(goal):
    cluster = goal['cluster']
    spend, save = predict_cluster[cluster]['spending'], predict_cluster[cluster]['saving']
    monthly_target = goal['target_amount'] / (goal['duration_days'] / 30)
    if goal['goal_type'] == 'saving':
        feasibility = min(save / monthly_target, 1.0) if monthly_target > 0 else 0.0
    else:  # spending
        feasibility = min(spend / monthly_target, 1.0) if monthly_target > 0 else 0.0
    return {
        'goal_type': 1 if goal['goal_type'] == 'saving' else 0,
        'goal_priority': {'Low': 0, 'Medium': 1, 'High': 2}[goal['goal_priority']],
        'goal_horizon': 1 if goal['goal_horizon'] == 'long' else 0,
        'target_amount': goal['target_amount'],
        'duration_days': goal['duration_days'], 
        'monthly_target': monthly_target,
        'feasibility': feasibility
    }

# Tạo vector đặc trưng cho tất cả mục tiêu
df_goals['features'] = df_goals.apply(extract_features, axis=1)

# Hàm tìm mục tiêu tương tự
def find_similar_goals(new_goal, df, top_k=5):
    # Tạo vector đặc trưng cho mục tiêu mới
    new_features = extract_features(new_goal)
    new_vector = np.array([
        new_features['goal_type'],
        new_features['goal_priority'],
        new_features['goal_horizon'],
        new_features['target_amount'],
        new_features['duration_days'],
        new_features['monthly_target'],
        new_features['feasibility']
    ]).reshape(1, -1)
    
    # Tạo vector đặc trưng cho các mục tiêu hiện có
    vectors = np.array([
        [
            f['goal_type'],
            f['goal_priority'],
            f['goal_horizon'],
            f['target_amount'],
            f['duration_days'],
            f['monthly_target'],
            f['feasibility']
        ] for f in df['features']
    ])
    
    # Tính độ tương đồng cosine
    similarities = cosine_similarity(new_vector, vectors)[0]
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Lấy các mục tiêu tương tự, bao gồm cột features
    similar_goals = df.iloc[top_k_indices][['goal_id', 'user_id', 'completion_percent', 'target_amount', 'duration_days', 'associated_jar', 'cluster', 'features']]
    
    return similar_goals

# Hàm đánh giá mục tiêu mới
def evaluate_new_goal(new_goal, df, clusters_df):
    # Gộp cluster cho mục tiêu mới
    user_id = new_goal['user_id']
    cluster = clusters_df[clusters_df['user_id'] == user_id]['cluster'].iloc[0] if user_id in clusters_df['user_id'].values else 0
    new_goal['cluster'] = cluster
    
    # Tính đặc trưng và feasibility cho mục tiêu mới
    new_features = extract_features(new_goal)
    new_feasibility = new_features['feasibility']
    
    # Tìm các mục tiêu tương tự
    similar_goals = find_similar_goals(new_goal, df)
    
    if similar_goals.empty:
        return {
            'feasibility': 'Không tìm thấy mục tiêu tương tự',
            'new_goal_feasibility': round(new_feasibility * 100, 2),
            'avg_completion': None,
            'avg_feasibility': None,
            'similar_goals': [],
            'suggestions': 'Kiểm tra lại thông tin mục tiêu hoặc thử với các tham số khác.'
        }
    
    # Tính trung bình completion_percent và feasibility của các mục tiêu tương tự
    avg_completion = similar_goals['completion_percent'].mean()
    avg_feasibility = similar_goals['features'].apply(lambda x: x['feasibility']).mean()
    
    # Xác định tính khả thi dựa trên cả completion_percent và feasibility
    feasibility_label = (
        'Khả thi' if avg_completion > 50 and avg_feasibility > 0.5 and new_feasibility > 0.5 else
        'Khó đạt' if (avg_completion > 20 or avg_feasibility > 0.2) and new_feasibility > 0.2 else
        'Rất khó đạt'
    )
    
    # Rule-based suggestions
    suggestions = []
    monthly_target = new_features['monthly_target']
    spend, save = predict_cluster[cluster]['spending'], predict_cluster[cluster]['saving']
    
    if new_goal['goal_type'] == 'saving':
        if monthly_target > save:
            suggestions.append(f"Số tiền tiết kiệm hàng tháng ({monthly_target:,.0f}) vượt khả năng tiết kiệm ({save:,.0f}). Cân nhắc giảm target_amount hoặc kéo dài target_date.")
        if new_feasibility < 0.5:
            suggestions.append(f"Khả năng đạt mục tiêu tiết kiệm thấp ({new_feasibility*100:.2f}%). Cân nhắc điều chỉnh mục tiêu phù hợp với khả năng tài chính.")
    elif new_goal['goal_type'] == 'spending':
        if monthly_target > spend:
            suggestions.append(f"Số tiền chi tiêu hàng tháng ({monthly_target:,.0f}) vượt khả năng chi tiêu ({spend:,.0f}). Cân nhắc giảm target_amount.")
        if new_feasibility < 0.5:
            suggestions.append(f"Khả năng đạt mục tiêu chi tiêu thấp ({new_feasibility*100:.2f}%). Cân nhắc điều chỉnh mục tiêu phù hợp với khả năng tài chính.")
    
    return {
        'feasibility': feasibility_label,
        'new_goal_feasibility': round(new_feasibility * 100, 2),
        'avg_completion': round(avg_completion, 2) if avg_completion is not None else None,
        'avg_feasibility': round(avg_feasibility * 100, 2),
        'similar_goals': similar_goals[['goal_id', 'user_id', 'completion_percent', 'target_amount', 'duration_days', 'associated_jar', 'cluster']].to_dict('records'),
        'suggestions': suggestions
    }

# Ví dụ mục tiêu mới
new_goal = {
    'user_id': 6,
    'goal_type': 'saving',
    'goal_priority': 'Medium',
    'goal_horizon': 'long',
    'target_amount': 1000000000,
    'start_date': pd.to_datetime('2025-07-01'),
    'target_date': pd.to_datetime('2028-07-01'),
    'associated_jar': 'EMERGENCY',
    'duration_days': (pd.to_datetime('2028-07-01') - pd.to_datetime('2025-07-01')).days
}

# Đánh giá mục tiêu
result = evaluate_new_goal(new_goal, df_goals, df_clusters)
print("Đánh giá mục tiêu mới:")
print(f"Tính khả thi: {result['feasibility']}")
print(f"Khả năng đạt mục tiêu mới: {result['new_goal_feasibility']}%")
print(f"Tỷ lệ hoàn thành trung bình của các mục tiêu tương tự: {result['avg_completion']}%")
print(f"Tỷ lệ khả thi trung bình: {result['avg_feasibility']}%")
print("Các mục tiêu tương tự:")
for goal in result['similar_goals']:
    print(f"- Goal ID: {goal['goal_id']}, User ID: {goal['user_id']}, Completion: {goal['completion_percent']}%, Jar: {goal['associated_jar']}, Cluster: {goal['cluster']}")
print("Gợi ý:")
for suggestion in result['suggestions']:
    print(f"- {suggestion}")