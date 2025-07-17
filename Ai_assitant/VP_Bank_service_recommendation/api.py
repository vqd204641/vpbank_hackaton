from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal
import uvicorn
import sys
import os

# Add project root to path to allow module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

from Ai_assitant.goals_recommendation.Difficulty_Rating.goals_evaluate import GoalEvaluator
from Ai_assitant.VP_Bank_service_recommendation.rag_utils import get_vectorstore, query_rag
from Ai_assitant.Jars_proportion_distribution.rules_base_recommendation import suggest_percents_for_user

app = FastAPI(
    title="VPBank AI Services API",
    description="An API for VPBank's AI-powered services, including service recommendations and goal evaluation.",
    version="1.1.0"
)

# --- Models for Request Bodies ---
class QueryRequest(BaseModel):
    query: str

class GoalRequest(BaseModel):
    user_id: int
    goal_type: Literal['saving', 'spending']
    goal_priority: Literal['Low', 'Medium', 'High']
    goal_horizon: Literal['long', 'short']
    target_amount: float
    start_date: str
    target_date: str
    associated_jar: str

class JarSuggestionRequest(BaseModel):
    income: float
    user_jars: list[str]
    current_percents: dict[str, float] | None = None

# --- Global Variables for Models ---
vectorstore = None
goal_evaluator = None

# --- Application Startup Event ---
@app.on_event("startup")
async def startup_event():
    """
    Initialize models and other resources on application startup.
    """
    global vectorstore, goal_evaluator
    
    try:
        vectorstore = get_vectorstore()
    except Exception as e:
        vectorstore = None
    try:
        goal_evaluator = GoalEvaluator()
    except Exception as e:
        goal_evaluator = None

# --- API Endpoints ---
@app.post("/query/", tags=["Service Recommendation"])
async def query_endpoint(request: QueryRequest):
    """
    Provides VPBank service recommendations based on a user query.
    """
    if vectorstore is None:
        raise HTTPException(status_code=503, detail="Vector store is not available. Please check server logs.")
    
    try:
        results = query_rag(request.query, vectorstore)
        if not results:
            return {"message": "No relevant documents found for your query.", "data": []}
        return {"message": "Query successful", "data": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during the query: {e}")

@app.post("/evaluate_goal/", tags=["Goal Evaluation"])
async def evaluate_goal_endpoint(request: GoalRequest):
    """
    Evaluates the feasibility of a new financial goal for a user.
    """
    if goal_evaluator is None:
        raise HTTPException(status_code=503, detail="Goal Evaluator is not available. Please check server logs.")

    try:
        # Convert Pydantic model to dictionary
        new_goal = request.dict()
        result = goal_evaluator.evaluate_new_goal(new_goal)
        return {"message": "Goal evaluation successful", "data": result}
    except Exception as e:
        # Log the exception for debugging
        print(f"Error during goal evaluation: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during goal evaluation: {e}")

@app.post("/suggest_jar_percents/", tags=["Jar Recommendation"])
async def suggest_jar_percents_endpoint(request: JarSuggestionRequest):
    """
    Suggests jar distribution percentages based on user income and existing jars.
    """
    try:
        suggestions = suggest_percents_for_user(
            income=request.income,
            user_jars=request.user_jars,
            percent_dict=request.current_percents
        )
        # Convert DataFrame to a list of dictionaries for JSON response
        return {"message": "Jar percentage suggestion successful", "data": suggestions}
    except Exception as e:
        # Log the exception for debugging
        print(f"Error during jar suggestion: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during jar suggestion: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
