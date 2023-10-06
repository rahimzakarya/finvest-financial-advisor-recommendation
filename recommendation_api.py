from fastapi import FastAPI, HTTPException
import openai
import pandas as pd
import utils

# Create a FastAPI app
app = FastAPI()

# API key setup
api_key = "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"
openai.api_key = api_key

# Load the DataFrame
df = pd.read_csv('./data/financial_habits_dataset.csv')

@app.post("/recommendations/")
async def get_recommendations(user_info: dict):
    try:
        # Get recommendations
        recommendations = utils.recommend_habits_knn(df, user_info)
        
        # Generate recommendations in context
        recommendations_in_context = utils.generate_recommendations(user_info, recommendations)
        
        return {"recommendations": recommendations_in_context}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))