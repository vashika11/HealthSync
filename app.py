from flask import Flask, render_template, request, redirect, url_for, flash, session
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import json
from models.nutrition_model import NutritionRecommender
from utils.data_processor import DataProcessor
from utils.validators import validate_user_input
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

# Initialize components
recommender = NutritionRecommender()
data_processor = DataProcessor()

# Session data structure
def init_session():
    if 'user_data' not in session:
        session['user_data'] = {
            'name': 'Guest User',
            'goals': {
                'calories': 2000,
                'protein': 150,
                'carbs': 250,
                'fat': 70,
                'fiber': 25
            },
            'current_nutrition': {
                'calories': 0,
                'protein': 0,
                'carbs': 0,
                'fat': 0,
                'fiber': 0
            },
            'meal_history': [],
            'preferences': []
        }

@app.before_request
def before_request():
    init_session()

@app.route('/')
def index():
    user_data = session.get('user_data', {})
    
    # Calculate progress percentages
    goals = user_data.get('goals', {})
    current = user_data.get('current_nutrition', {})
    
    progress = {}
    for nutrient in ['calories', 'protein', 'carbs', 'fat', 'fiber']:
        goal_value = goals.get(nutrient, 1)
        current_value = current.get(nutrient, 0)
        progress[nutrient] = min(100, (current_value / goal_value) * 100) if goal_value > 0 else 0
    
    # Get today's meal suggestions
    suggestions = recommender.get_daily_suggestions(goals)
    
    return render_template('index.html', 
                         user_data=user_data, 
                         progress=progress,
                         suggestions=suggestions)

@app.route('/meal_planner', methods=['GET', 'POST'])
def meal_planner():
    if request.method == 'POST':
        # Process form data
        form_data = {
            'calories': int(request.form.get('calories', 2000)),
            'protein': int(request.form.get('protein', 150)),
            'carbs': int(request.form.get('carbs', 250)),
            'fat': int(request.form.get('fat', 70)),
            'activity_level': request.form.get('activity_level', 'moderate'),
            'dietary_restrictions': request.form.getlist('dietary_restrictions'),
            'meal_count': int(request.form.get('meal_count', 3))
        }
        
        if validate_user_input(form_data):
            # Update session with new goals
            session['user_data']['goals'] = form_data
            
            # Generate meal plan
            meal_plan = recommender.generate_weekly_meal_plan(form_data)
            
            # Calculate nutrition totals
            nutrition_summary = data_processor.calculate_weekly_nutrition(meal_plan)
            
            return render_template('results.html', 
                                 meal_plan=meal_plan, 
                                 nutrition_summary=nutrition_summary,
                                 user_goals=form_data)
        else:
            flash('Please check your input values and try again.', 'error')
    
    return render_template('meal_planner.html')

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if request.method == 'POST':
        # Update user profile
        session['user_data']['name'] = request.form.get('name', 'Guest User')
        
        # Update preferences
        preferences = request.form.getlist('food_preferences')
        session['user_data']['preferences'] = preferences
        
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('profile'))
    
    return render_template('profile.html', user_data=session.get('user_data', {}))

@app.route('/log_meal', methods=['POST'])
def log_meal():
    meal_type = request.form.get('meal_type')
    food_items = request.form.get('food_items', '').split(',')
    portions = request.form.get('portions', '100')  # grams
    
    try:
        portions = float(portions)
        nutrition = data_processor.calculate_meal_nutrition(food_items, portions)
        
        # Update current nutrition in session
        current = session['user_data']['current_nutrition']
        for nutrient, value in nutrition.items():
            current[nutrient] = current.get(nutrient, 0) + value
        
        # Add to meal history
        meal_entry = {
            'type': meal_type,
            'foods': food_items,
            'nutrition': nutrition,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time': datetime.now().strftime('%H:%M')
        }
        session['user_data']['meal_history'].append(meal_entry)
        
        flash(f'{meal_type.title()} logged successfully!', 'success')
    except Exception as e:
        flash(f'Error logging meal: {str(e)}', 'error')
    
    return redirect(url_for('index'))

@app.route('/reset_day')
def reset_day():
    session['user_data']['current_nutrition'] = {
        'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0, 'fiber': 0
    }
    session['user_data']['meal_history'] = []
    flash('Daily nutrition reset!', 'info')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)