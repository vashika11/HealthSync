# app.py
from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)
df = pd.read_csv('data/nutrition.csv')
model = pickle.load(open('model/diet_model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/plan', methods=['POST'])
def plan():
    calories = float(request.form['calories'])
    protein = float(request.form['protein'])
    fat = float(request.form['fat'])
    carbs = float(request.form['carbs'])
    fiber = float(request.form['fiber'])

    input_data = [[calories, protein, fat, carbs, fiber]]
    indices = model.kneighbors(input_data, return_distance=False)[0]
    recommendations = df.iloc[indices].to_dict(orient='records')

    return render_template('meal_plan.html', meals=recommendations)

if __name__ == '__main__':
    app.run(debug=True)