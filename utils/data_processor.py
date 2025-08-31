import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataProcessor:
    def __init__(self):
        self.nutrition_database = None
        self.load_nutrition_data()
    
    def load_nutrition_data(self):
        try:
            self.nutrition_database = pd.read_csv('data/nutrition.csv')
        except FileNotFoundError:
            print("Nutrition database not found!")
    
    def calculate_meal_nutrition(self, food_items, portion_grams=100):
        """Calculate nutrition for a meal with given foods and portion"""
        total_nutrition = {
            'calories': 0,
            'protein': 0,
            'carbs': 0,
            'fat': 0,
            'fiber': 0
        }
        
        for food_name in food_items:
            food_name = food_name.strip()
            food_data = self.nutrition_database[
                self.nutrition_database['food_name'].str.lower() == food_name.lower()
            ]
            
            if not food_data.empty:
                food = food_data.iloc[0]
                multiplier = portion_grams / 100.0
                
                total_nutrition['calories'] += food['calories_per_100g'] * multiplier
                total_nutrition['protein'] += food['protein'] * multiplier
                total_nutrition['carbs'] += food['carbs'] * multiplier
                total_nutrition['fat'] += food['fat'] * multiplier
                total_nutrition['fiber'] += food['fiber'] * multiplier
        
        return total_nutrition
    
    def calculate_weekly_nutrition(self, weekly_plan):
        """Calculate total nutrition for a weekly meal plan"""
        weekly_totals = {
            'calories': 0,
            'protein': 0,
            'carbs': 0,
            'fat': 0,
            'fiber': 0
        }
        
        daily_averages = []
        
        for day, meals in weekly_plan.items():
            daily_total = {
                'calories': 0,
                'protein': 0,
                'carbs': 0,
                'fat': 0,
                'fiber': 0
            }
            
            for meal_type, foods in meals.items():
                for food in foods:
                    daily_total['calories'] += food['calories_per_100g']
                    daily_total['protein'] += food['protein']
                    daily_total['carbs'] += food['carbs']
                    daily_total['fat'] += food['fat']
                    daily_total['fiber'] += food['fiber']
            
            daily_averages.append(daily_total)
            
            for nutrient in weekly_totals:
                weekly_totals[nutrient] += daily_total[nutrient]
        
        # Calculate averages
        for nutrient in weekly_totals:
            weekly_totals[f'avg_{nutrient}'] = weekly_totals[nutrient] / 7
        
        return {
            'weekly_totals': weekly_totals,
            'daily_breakdown': dict(zip(weekly_plan.keys(), daily_averages))
        }
    
    def get_food_by_category(self, category):
        """Get all foods in a specific category"""
        if self.nutrition_database is not None:
            return self.nutrition_database[
                self.nutrition_database['category'].str.lower() == category.lower()
            ].to_dict('records')
        return []
    
    def search_foods(self, query):
        """Search for foods by name"""
        if self.nutrition_database is not None:
            mask = self.nutrition_database['food_name'].str.contains(
                query, case=False, na=False
            )
            return self.nutrition_database[mask].to_dict('records')
        return []