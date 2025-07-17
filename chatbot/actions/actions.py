import requests
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

class ActionSuggestJarAllocation(Action):
    def name(self) -> Text:
        return "action_suggest_jar_allocation"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # user_message = tracker.latest_message.get("text")
        try:
            response = requests.post(
                "http://localhost:8000/suggest_jar_percents",
                json={"income": 15000000,"user_jars": ["NEC", "EDU", "FFA", "PLAY", "LTSS"]},
                timeout=5
            )
            suggestions = response.json().get("data", {})
        except Exception as e:
            result = f"Đã xảy ra lỗi khi gọi API: {str(e)}"
        # print(suggestions)
        dispatcher.utter_message(text=suggestions)
        return []

class ActionEvaluateFinancialGoal(Action):
    def name(self) -> Text:
        return "action_evaluate_financial_goal"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        user_message = tracker.latest_message.get("text")
        try:
            response = requests.post(
                "http://localhost:8000/evaluate_goal",
                json={
                    'user_id': 6,
                    'goal_type': 'saving',
                    'goal_priority': 'Medium',
                    'goal_horizon': 'long',
                    'target_amount': 10000000,
                    'start_date': '2025-07-01',
                    'target_date': '2028-07-01',
                    'associated_jar': 'EMERGENCY',
                },
                timeout=5
            )
            suggestions = response.json().get("data", {})
        except Exception as e:
            result = f"Đã xảy ra lỗi khi gọi API: {str(e)}"
        # print(suggestions)
        dispatcher.utter_message(text=suggestions)
        return []


class ActionSuggestVPBankService(Action):
    def name(self) -> Text:
        return "action_suggest_vpbank_service"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        user_message = tracker.latest_message.get("text")

        try:
            response = requests.post(
                "http://localhost:8000/query",
                json={"query": user_message},
                timeout=5
            )
            suggestions = response.json().get("data", {})
        except Exception as e:
            result = f"Đã xảy ra lỗi khi gọi API: {str(e)}"
        dispatcher.utter_message(text=suggestions)
        return []

class ActionCallRemoteFunc(Action):

    def name(self) -> Text:
        return "action_call_remote_func"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        dispatcher.utter_message(text="test thành công")
        return []