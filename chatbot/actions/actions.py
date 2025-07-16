from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

# class ActionSuggestJarAllocation(Action):
#     def name(self) -> Text:
#         return "action_suggest_jar_allocation"

#     def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#         # Simulate logic for jar allocation suggestion
#         user_income = tracker.get_slot("income") or "unknown"
#         current_jars = tracker.get_slot("jar") or "unknown"
#         # Example logic: Suggest fixed allocation based on description
#         suggested_allocation = "55% nhu cầu thiết yếu, 7% tiết kiệm dài hạn, 13% đầu tư, 12.5% giáo dục, 12.5% hưởng thụ, 0% từ thiện"
        
#         dispatcher.utter_message(response="utter_suggest_jar_allocation", jar_allocation=suggested_allocation)
#         return []

class ActionEvaluateGoal(Action):
    def name(self) -> Text:
        return "action_evaluate_goal"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Extract goal and related slots
        goal = tracker.get_slot("goal") or "unknown"
        time = tracker.get_slot("time") or "unknown"
        amount = tracker.get_slot("amount") or "unknown"
        
        # Simulate goal evaluation logic (e.g., Monte Carlo + fuzzy logic)
        if "mua nhà" in goal and "3 năm" in time:
            difficulty = "rất khó"
            suggested_goal = "mua nhà trong 5 năm hoặc tăng tiết kiệm 10 triệu/tháng"
        else:
            difficulty = "phù hợp"
            suggested_goal = f"đạt {goal} trong {time}"
        
        dispatcher.utter_message(response="utter_suggest_goal", goal=goal, difficulty=difficulty, suggested_goal=suggested_goal)
        return []

# class ActionSuggestVPBankService(Action):
#     def name(self) -> Text:
#         return "action_suggest_vpbank_service"

#     def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#         # Extract service slot
#         service = tracker.get_slot("service") or "unknown"
        
#         # Simulate semantic search for VPBank services
#         service_data = {
#             "tiết kiệm online": {
#                 "name": "Tiết kiệm Online VPBank",
#                 "benefits": "Lãi suất cao, gửi tiền linh hoạt",
#                 "link": "https://vpbank.com.vn/tiet-kiem-online"
#             },
#             "vay vốn": {
#                 "name": "Vay Vốn VPBank",
#                 "benefits": "Hạn mức cao, thủ tục nhanh gọn",
#                 "link": "https://vpbank.com.vn/vay-von"
#             },
#             "thẻ tín dụng": {
#                 "name": "Thẻ Tín Dụng VPBank",
#                 "benefits": "Ưu đãi hoàn tiền, tích điểm",
#                 "link": "https://vpbank.com.vn/the-tin-dung"
#             },
#             "tài khoản số đẹp": {
#                 "name": "Tài Khoản Số Đẹp VPBank",
#                 "benefits": "Số tài khoản tùy chọn, dễ nhớ",
#                 "link": "https://vpbank.com.vn/tai-khoan-so-dep"
#             }
#         }
        
#         service_info = service_data.get(service, {
#             "name": "Dịch vụ VPBank",
#             "benefits": "Phù hợp với nhu cầu tài chính của bạn",
#             "link": "https://vpbank.com.vn"
#         })
        
#         dispatcher.utter_message(
#             response="utter_suggest_service",
#             service=service_info["name"],
#             benefits=service_info["benefits"],
#             link=service_info["link"]
#         )
#         return []