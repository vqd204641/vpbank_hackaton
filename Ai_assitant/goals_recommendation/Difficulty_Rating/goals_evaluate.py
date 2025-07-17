import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import sys
import os

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

project_root = os.path.abspath(os.path.join(os.getcwd(), "../../../"))

# Adjust the path to be relative to this file's location
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

from spending_forecast.forecast import ARIMAPredictor

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv(dotenv_path="../../.env")
# Lấy API key từ biến môi trường


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

        api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash-latest")

        # Preprocess data
        self._preprocess_data()

    def calculate_feasibility_label(self, avg_feasibility, new_feasibility, goal_priority, goal_amount):
        avg_feas = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'avg_feasibility')
        new_feas = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'new_feasibility')
        priority = ctrl.Antecedent(np.arange(0, 3, 1), 'goal_priority')
        amount = ctrl.Antecedent(np.arange(0, 5_000_000_001, 100_000_000), 'goal_amount')

        feasibility = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'feasibility_label')

        # Membership functions
        avg_feas.automf(3)
        new_feas.automf(3)
        priority['low'] = fuzz.trimf(priority.universe, [0, 0, 1])
        priority['medium'] = fuzz.trimf(priority.universe, [0, 1, 2])
        priority['high'] = fuzz.trimf(priority.universe, [1, 2, 2])

        amount['small'] = fuzz.trimf(amount.universe, [0, 0, 1_000_000_000])
        amount['medium'] = fuzz.trimf(amount.universe, [500_000_000, 2_000_000_000, 3_000_000_000])
        amount['large'] = fuzz.trimf(amount.universe, [2_000_000_000, 5_000_000_000, 5_000_000_000])

        feasibility['low'] = fuzz.trimf(feasibility.universe, [0, 0, 0.5])
        feasibility['medium'] = fuzz.trimf(feasibility.universe, [0.3, 0.5, 0.7])
        feasibility['high'] = fuzz.trimf(feasibility.universe, [0.5, 1, 1])

        # Labels for easier iteration
        avg_levels = ['poor', 'average', 'good']      # auto-named by automf
        new_levels = ['poor', 'average', 'good']
        priority_levels = ['low', 'medium', 'high']
        amount_levels = ['small', 'medium', 'large']

        rules = []
        for af in avg_levels:
            for nf in new_levels:
                for pr in priority_levels:
                    for am in amount_levels:
                        # Rule logic
                        if af == 'good' and nf == 'good' and pr == 'low' and am in ['small', 'medium']:
                            out = 'high'
                        elif af == 'poor' or nf == 'poor' or pr == 'high' or am == 'large':
                            out = 'low'
                        else:
                            out = 'medium'

                        rule = ctrl.Rule(
                            avg_feas[af] & new_feas[nf] & priority[pr] & amount[am],
                            feasibility[out]
                        )
                        rules.append(rule)

        # Build system
        system = ctrl.ControlSystem(rules)
        sim = ctrl.ControlSystemSimulation(system)

        sim.input['avg_feasibility'] = float(avg_feasibility)
        sim.input['new_feasibility'] = float(new_feasibility)
        sim.input['goal_priority'] = float(goal_priority)
        sim.input['goal_amount'] = float(goal_amount)

        sim.compute()
        return round(sim.output['feasibility_label'], 3)
    
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
        goal_priority = {'Low': 0, 'Medium': 1, 'High': 2}.get(new_features['goal_priority'], 1)
        goal_amount = new_features["target_amount"]
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

        feasibility_label = self.calculate_feasibility_label(avg_feasibility=avg_feasibility,new_feasibility=new_feasibility,goal_priority=goal_priority,goal_amount=goal_amount)

        suggestions = []
        monthly_target = new_features['monthly_target']
        spend, save = self.predict_cluster[cluster]['spending'], self.predict_cluster[cluster]['saving']
        prompt = f"""
        Bạn là một chuyên gia tài chính cá nhân. Hãy phân tích và đưa ra đánh giá cho mục tiêu tài chính sau của người dùng.

        ### Thông tin mục tiêu:
        - Mức độ ưu tiên của mục tiêu: {new_features['goal_priority']} (mã số: {goal_priority})
        - Số tiền mục tiêu: {goal_amount:,} VNĐ
        - Số ngày thực hiện: {new_goal['duration_days']} ngày
        - Số tiền cần tiết kiệm mỗi tháng để đạt mục tiêu: {monthly_target:,.0f} VNĐ/tháng
        - Tỷ lệ khả thi của mục tiêu (tính toán bởi hệ thống): {new_feasibility:.2f} (trên thang 0–1)
        - Mức độ khả thi trung bình của các mục tiêu tương tự: {avg_feasibility:.2f}
        - Tỷ lệ hoàn thành trung bình của các mục tiêu tương tự: {avg_completion:.2f}%
        - Nhãn đánh giá mức độ khả thi được tính toán: {feasibility_label*100}% khả năng hoàn thành

        ### Yêu cầu:
        1. Trình bày ngắn gọn trong 6-8 câu
        2. Đưa ra đánh giá tổng quan về khả năng đạt mục tiêu của người dùng dựa trên các thông tin trên.
        3. Nếu mục tiêu có rủi ro không đạt, gợi ý điều chỉnh cụ thể (ví dụ: kéo dài thời gian, giảm số tiền, tăng tiết kiệm,...).
        4. Trình bày bằng tiếng Việt, rõ ràng, có phân tích logic.
        """
        response = self.model .generate_content(prompt)
        return response.text
    
if __name__ == '__main__':
    evaluator = GoalEvaluator()
    
    new_goal_example = {
        'user_id': 6,
        'goal_type': 'saving',
        'goal_priority': 'Medium',
        'goal_horizon': 'long',
        'target_amount': 10000000,
        'start_date': '2025-07-01',
        'target_date': '2028-07-01',
        'associated_jar': 'EMERGENCY',
    }
    
    result = evaluator.evaluate_new_goal(new_goal_example)
    print(result)
