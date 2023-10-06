import pandas as pd
from sklearn.neighbors import NearestNeighbors
import openai
import json

def recommend_habits_knn(df, user_info, num_recommendations=3):
    """
    Recommends habits to a user based on their financial behavior scores using k-Nearest Neighbors.

    Parameters:
    - df (DataFrame): A pandas DataFrame containing financial behavior data for multiple users.
    - user_id (int): The user ID for which recommendations are generated.
    - num_recommendations (int): The number of habit recommendations to return (default is 3).

    Returns:
    - recommended_habits (list of lists): A list of recommended habits for the user.
    """

    # Create a DataFrame from the Json 
    print(type(user_info))
    user_info_df = pd.DataFrame(user_info, index=[0])


    # Extract the user's financial behavior scores
    user_scores = user_info_df[[
        'Savings_Points', 'Budget_Adherence_Points', 'Expense_Tracking_Points',
        'Timely_Bill_Payments_Points', 'Avoiding_Unnecessary_Expenses_Points',
        'Goal_Progress_Points', 'Debt_Management_Points',
        'Financial_Education_Points', 'Regular_App_Usage_Points',
        'Security_Privacy_Points', 'Community_Social_Points'
    ]].values

    # Check if the user was not found
    if len(user_scores) == 0:
        return []

    # Extract the relevant columns for k-NN
    features = df[[
        'Savings_Points', 'Budget_Adherence_Points', 'Expense_Tracking_Points',
        'Timely_Bill_Payments_Points', 'Avoiding_Unnecessary_Expenses_Points',
        'Goal_Progress_Points', 'Debt_Management_Points',
        'Financial_Education_Points', 'Regular_App_Usage_Points',
        'Security_Privacy_Points', 'Community_Social_Points'
    ]].values

    # Create a k-NN model
    knn = NearestNeighbors(n_neighbors=num_recommendations, metric='cosine')
    knn.fit(features)

    # Find the k nearest neighbors to the user's scores
    distances, indices = knn.kneighbors(user_scores)

    # Get the indices of the k nearest neighbors
    neighbor_indices = indices[0]

    # Exclude the user's own row
    neighbor_indices = [
        idx for idx in neighbor_indices
    ]

    # Extract recommended habits from the top neighbors
    recommended_habits = df.iloc[neighbor_indices][[
        'Recommended_Habits_1', 'Recommended_Habits_2',
        'Recommended_Habits_3', 'Recommended_Habits_4'
    ]].values.tolist()

    return recommended_habits

def generate_recommendations(user_info, recommendations):
    """
    Generate financial recommendations based on the user's information and the recommendations from a previous model.

    Parameters:
    - user_info (str): A string containing the user's information.
    - recommendations (list of str): A list of financial recommendations to include in the prompt.

    Returns:
    - generated_recommendations (list of str): A list of generated recommendations.
    """

    # Define the prompt for the OpenAI API request
    prompt = f"I am a financial advisor, I advice you to {recommendations[0]}, {recommendations[1]},{recommendations[2]}, , with the following information:\n\n{user_info}\n\n tell him you why, and it will impact in your financial health on the long and short term, and also using the user numbers, \n\nRecommended habits:\n"

    # Make a request to the OpenAI API
    response = openai.Completion.create(
        engine="text-davinci-002",  # You can choose the appropriate engine
        prompt=prompt,
        max_tokens=150,  # Adjust this based on the desired length of recommendations
        n = 5 # Number of recommendations to generate
    )

    # Extract and return the recommendations
    recommendations = response.choices[0].text.strip().split('\n')
    return recommendations

