import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import sys
import os
project_root = os.path.abspath(os.path.join(os.getcwd(), "../../../"))

# Adjust the path to be relative to this file's location
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

from spending_forecast.forecast import ARIMAPredictor

class GoalEvaluator:
    def __init__(self):
        # Load data and initialize models once
        self.predictor = ARIMAPredictor()
        self.predict_cluster = {}
        for i in range(4):
            result = self.predictor.predict_cluster_spending(i)
            spending = result.get('EDU', 0) + result.get('PLAY', 0)
            saving = result.get('FFA', 0) + result.get('LTSS', 0)
            self.predict_cluster[i] = {"spending": spending, "saving": saving}

        # Define paths relative to the project root
        user_data_path = os.path.join(project_root, "user_data_ver1/user_goals_2saving_1spending.csv")
        clusters_path = os.path.join(project_root, "user_classification/user_clusters.csv")

        self.df_goals = pd.read_csv(user_data_path)
        self.df_clusters = pd.read_csv(clusters_path)

        # Preprocess data
        self._preprocess_data()

    def _preprocess_data(self):
        self.df_goals['start_date'] = pd.to_datetime(self.df_goals['start_date'])
        self.df_goals['target_date'] = pd.to_datetime(self.df_goals['target_date'])
        self.df_goals['duration_days'] = (self.df_goals['target_date'] - self.df_goals['start_date']).dt.days
        self.df_goals = self.df_goals.merge(self.df_clusters[['user_id', 'cluster']], on='user_id', how='left')
        self.df_goals['cluster'] = self.df_goals['cluster'].fillna(0).astype(int)
        self.df_goals['features'] = self.df_goals.apply(self._extract_features, axis=1)

    def _extract_features(self, goal):
        cluster = goal.get('cluster', 0)
        spend, save = self.predict_cluster.get(cluster, {}).get('spending', 0), self.predict_cluster.get(cluster, {}).get('saving', 0)
        
        duration_days = goal.get('duration_days', 1)
        if duration_days <= 0:
            duration_days = 1

        monthly_target = goal['target_amount'] / (duration_days / 30)
        
        if goal['goal_type'] == 'saving':
            feasibility = min(save / monthly_target, 1.0) if monthly_target > 0 else 0.0
        else:  # spending
            feasibility = min(spend / monthly_target, 1.0) if monthly_target > 0 else 0.0
            
        return {
            'goal_type': 1 if goal['goal_type'] == 'saving' else 0,
            'goal_priority': {'Low': 0, 'Medium': 1, 'High': 2}.get(goal['goal_priority'], 1),
            'goal_horizon': 1 if goal['goal_horizon'] == 'long' else 0,
            'target_amount': goal['target_amount'],
            'duration_days': duration_days,
            'monthly_target': monthly_target,
            'feasibility': feasibility
        }

    def _find_similar_goals(self, new_goal_features, top_k=5):
        new_vector = np.array(list(new_goal_features.values())).reshape(1, -1)

        vectors = np.array([list(f.values()) for f in self.df_goals['features']])

        similarities = cosine_similarity(new_vector, vectors)[0]
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return self.df_goals.iloc[top_k_indices]

    def evaluate_new_goal(self, new_goal):
        # Ensure dates are in datetime format
        new_goal['start_date'] = pd.to_datetime(new_goal['start_date'])
        new_goal['target_date'] = pd.to_datetime(new_goal['target_date'])
        new_goal['duration_days'] = (new_goal['target_date'] - new_goal['start_date']).days

        user_id = new_goal['user_id']
        if user_id in self.df_clusters['user_id'].values:
            cluster = self.df_clusters[self.df_clusters['user_id'] == user_id]['cluster'].iloc[0]
        else:
            cluster = 0
        new_goal['cluster'] = int(cluster)

        new_features = self._extract_features(new_goal)
        new_feasibility = new_features['feasibility']

        similar_goals = self._find_similar_goals(new_features)

        if similar_goals.empty:
            return {
                'feasibility': 'Không tìm thấy mục tiêu tương tự',
                'new_goal_feasibility': round(new_feasibility * 100, 2),
                'avg_completion': None,
                'avg_feasibility': None,
                'similar_goals': [],
                'suggestions': 'Kiểm tra lại thông tin mục tiêu hoặc thử với các tham số khác.'
            }

        avg_completion = similar_goals['completion_percent'].mean()
        avg_feasibility = similar_goals['features'].apply(lambda x: x['feasibility']).mean()

        feasibility_label = (
            'Khả thi' if avg_completion > 50 and avg_feasibility > 0.5 and new_feasibility > 0.5 else
            'Khó đạt' if (avg_completion > 20 or avg_feasibility > 0.2) and new_feasibility > 0.2 else
            'Rất khó đạt'
        )

        suggestions = []
        monthly_target = new_features['monthly_target']
        spend, save = self.predict_cluster[cluster]['spending'], self.predict_cluster[cluster]['saving']

        if new_goal['goal_type'] == 'saving':
            if monthly_target > save and save > 0:
                suggestions.append(f"Số tiền tiết kiệm hàng tháng ({monthly_target:,.0f}) vượt khả năng tiết kiệm ({save:,.0f}). Cân nhắc giảm target_amount hoặc kéo dài target_date.")
            if new_feasibility < 0.5:
                suggestions.append(f"Khả năng đạt mục tiêu tiết kiệm thấp ({new_feasibility*100:.2f}%). Cân nhắc điều chỉnh mục tiêu.")
        elif new_goal['goal_type'] == 'spending':
            if monthly_target > spend and spend > 0:
                suggestions.append(f"Số tiền chi tiêu hàng tháng ({monthly_target:,.0f}) vượt khả năng chi tiêu ({spend:,.0f}). Cân nhắc giảm target_amount.")
            if new_feasibility < 0.5:
                suggestions.append(f"Khả năng đạt mục tiêu chi tiêu thấp ({new_feasibility*100:.2f}%). Cân nhắc điều chỉnh mục tiêu.")

        return {
            'feasibility': feasibility_label,
            'new_goal_feasibility': round(new_feasibility * 100, 2),
            #'avg_completion': round(avg_completion, 2) if pd.notna(avg_completion) else None,
            'avg_feasibility': round(avg_feasibility * 100, 2),
            #'similar_goals': similar_goals[['goal_id', 'user_id', 'completion_percent', 'target_amount', 'duration_days', 'associated_jar', 'cluster']].to_dict('records'),
            'suggestions': suggestions
        }

# if __name__ == '__main__':
#     evaluator = GoalEvaluator()
    
#     new_goal_example = {
#         'user_id': 6,
#         'goal_type': 'saving',
#         'goal_priority': 'Medium',
#         'goal_horizon': 'long',
#         'target_amount': 10000000,
#         'start_date': '2025-07-01',
#         'target_date': '2028-07-01',
#         'associated_jar': 'EMERGENCY',
#     }
    
#     result = evaluator.evaluate_new_goal(new_goal_example)
#     print(result)