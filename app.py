from flask import Flask, request
from flask_cors import CORS
import openai
import pandas as pd
import utils

# Create a Flask app
app = Flask(__name__)


# API key setup
api_key = "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"
openai.api_key = api_key

# Load the DataFrame
df = pd.read_csv('./data/financial_habits_dataset.csv')


@app.route("/")
def health():
    return "Welcome to Finvest recommendation system API"


@app.post("/recommendations")
def get_recommendations():
    features = {
        'Savings_Points': request.json['Savings_Points'],
        'Budget_Adherence_Points': request.json['Budget_Adherence_Points'],
        'Expense_Tracking_Points': request.json['Expense_Tracking_Points'],
        'Timely_Bill_Payments_Points': request.json['Timely_Bill_Payments_Points'],
        'Avoiding_Unnecessary_Expenses_Points': request.json['Avoiding_Unnecessary_Expenses_Points'],
        'Goal_Progress_Points': request.json['Goal_Progress_Points'],
        'Debt_Management_Points': request.json['Debt_Management_Points'],
        'Financial_Education_Points': request.json['Financial_Education_Points'],
        'Regular_App_Usage_Points': request.json['Regular_App_Usage_Points'],
        'Security_Privacy_Points': request.json['Security_Privacy_Points'],
        'Community_Social_Points': request.json['Community_Social_Points']
    }

    recommendations = utils.recommend_habits_knn(df, features)

    recommendations_in_context = utils.generate_recommendations(
        features, recommendations)
    
    cleaned_recommendations = [recommendation for recommendation in recommendations_in_context if recommendation]

    return {
        "data": cleaned_recommendations,
        "success": True
    }


if __name__ == "__main__":
    app.run(debug=True)