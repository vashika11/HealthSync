import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle
import os
from datetime import datetime, timedelta

class NutritionRecommender:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.food_database = None
        self.user_clusters = None
        self.load_or_create_data()
        self.load_or_train_model()
    
    def load_or_create_data(self):
        try:
            self.food_database = pd.read_csv('data/nutrition.csv')
        except FileNotFoundError:
            self.create_comprehensive_food_database()
    
    def create_comprehensive_food_database(self):
        # Comprehensive food database
        foods_data = {
            'food_name': [
                # Fruits
                'Apple', 'Banana', 'Orange', 'Strawberries', 'Blueberries', 'Grapes', 'Mango', 'Pineapple',
                # Vegetables
                'Broccoli', 'Spinach', 'Carrots', 'Bell Peppers', 'Tomatoes', 'Cucumber', 'Lettuce', 'Kale',
                # Proteins
                'Chicken Breast', 'Salmon', 'Tuna', 'Eggs', 'Turkey', 'Lean Beef', 'Tofu', 'Tempeh',
                # Grains
                'Brown Rice', 'Quinoa', 'Oats', 'Whole Wheat Bread', 'Barley', 'Bulgur', 'Millet',
                # Dairy
                'Greek Yogurt', 'Cottage Cheese', 'Milk', 'Cheddar Cheese', 'Mozzarella',
                # Nuts and Seeds
                'Almonds', 'Walnuts', 'Chia Seeds', 'Flax Seeds', 'Pumpkin Seeds',
                # Legumes
                'Black Beans', 'Chickpeas', 'Lentils', 'Kidney Beans',
                # Healthy Fats
                'Avocado', 'Olive Oil', 'Coconut Oil'
            ],
            'calories_per_100g': [
                52, 89, 47, 32, 57, 62, 60, 50,
                34, 23, 41, 31, 18, 12, 15, 35,
                165, 208, 144, 155, 189, 250, 70, 190,
                111, 368, 389, 265, 123, 342, 378,
                100, 98, 42, 113, 85,
                579, 654, 486, 534, 559,
                132, 164, 116, 127,
                160, 884, 862
            ],
            'protein': [
                0.3, 1.1, 0.9, 0.7, 0.7, 0.6, 0.8, 0.5,
                2.8, 2.9, 0.9, 1.9, 0.9, 0.7, 1.4, 2.9,
                31.0, 25.4, 30.0, 13.0, 29.0, 26.0, 8.1, 19.0,
                2.7, 14.1, 16.9, 13.0, 2.3, 12.5, 11.0,
                10.0, 11.1, 3.4, 25.0, 22.2,
                21.2, 15.2, 17.8, 18.3, 19.0,
                8.9, 8.0, 9.0, 9.3,
                2.0, 0.0, 0.0
            ],
            'carbs': [
                14, 23, 12, 8, 14, 16, 15, 13,
                7, 3.6, 10, 7, 3.9, 2.2, 2.9, 4.4,
                0, 0, 0, 1.1, 0, 0, 1.9, 9.0,
                23, 64, 66, 49, 28, 76, 73,
                3.6, 3.4, 5.0, 1.3, 2.2,
                22, 14, 42, 29, 15,
                20, 27, 20, 22,
                9, 0, 0
            ],
            'fat': [
                0.2, 0.3, 0.1, 0.3, 0.3, 0.2, 0.4, 0.1,
                0.4, 0.4, 0.2, 0.3, 0.2, 0.1, 0.2, 1.5,
                3.6, 13.4, 4.9, 11.0, 7.4, 15.0, 4.2, 11.0,
                0.9, 6.1, 6.9, 4.2, 1.2, 2.3, 4.2,
                0.4, 4.3, 1.0, 9.0, 6.1,
                49.9, 65.2, 30.7, 42.2, 49.1,
                0.5, 2.6, 0.4, 0.8,
                15.0, 100.0, 99.0
            ],
            'fiber': [
                2.4, 2.6, 2.4, 2.0, 2.4, 0.9, 1.6, 1.4,
                2.6, 2.2, 2.8, 3.4, 1.2, 0.5, 1.3, 4.1,
                0, 0, 0, 0, 0, 0, 1.9, 5.4,
                1.8, 7.0, 10.6, 6.8, 6.0, 18.3, 8.5,
                0, 0, 0, 0, 0,
                12.5, 6.7, 34.4, 27.3, 18.4,
                8.7, 8.0, 7.9, 7.4,
                6.7, 0, 0
            ],
            'category': [
                'Fruit', 'Fruit', 'Fruit', 'Fruit', 'Fruit', 'Fruit', 'Fruit', 'Fruit',
                'Vegetable', 'Vegetable', 'Vegetable', 'Vegetable', 'Vegetable', 'Vegetable', 'Vegetable', 'Vegetable',
                'Protein', 'Protein', 'Protein', 'Protein', 'Protein', 'Protein', 'Protein', 'Protein',
                'Grain', 'Grain', 'Grain', 'Grain', 'Grain', 'Grain', 'Grain',
                'Dairy', 'Dairy', 'Dairy', 'Dairy', 'Dairy',
                'Nuts', 'Nuts', 'Seeds', 'Seeds', 'Seeds',
                'Legume', 'Legume', 'Legume', 'Legume',
                'Healthy Fat', 'Healthy Fat', 'Healthy Fat'
            ],
            'meal_type': [
                'breakfast', 'breakfast', 'breakfast', 'breakfast', 'breakfast', 'any', 'any', 'any',
                'lunch', 'lunch', 'lunch', 'lunch', 'lunch', 'any', 'any', 'lunch',
                'lunch', 'dinner', 'lunch', 'breakfast', 'dinner', 'dinner', 'lunch', 'dinner',
                'lunch', 'any', 'breakfast', 'breakfast', 'lunch', 'any', 'breakfast',
                'breakfast', 'breakfast', 'breakfast', 'any', 'any',
                'snack', 'snack', 'breakfast', 'breakfast', 'snack',
                'lunch', 'lunch', 'lunch', 'lunch',
                'any', 'any', 'any'
            ]
        }
        
        df = pd.DataFrame(foods_data)
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/nutrition.csv', index=False)
        self.food_database = df

    def load_or_train_model(self):
        model_file = 'models/nutrition_model.pkl'
        scaler_file = 'models/scaler.pkl'
        
        try:
            with open(model_file, 'rb') as f:
                self.model = pickle.load(f)
            with open(scaler_file, 'rb') as f:
                self.scaler = pickle.load(f)
        except FileNotFoundError:
            self.train_model()
    
    def train_model(self):
        # Prepare features for ML model
        features = ['calories_per_100g', 'protein', 'carbs', 'fat', 'fiber']
        X = self.food_database[features]
        
        # Create target variable (nutritional quality score)
        y = (
            self.food_database['protein'] * 0.25 +
            self.food_database['fiber'] * 0.20 +
            (self.food_database['calories_per_100g'] / 100) * -0.15 +
            np.random.normal(0, 0.1, len(self.food_database))  # Add some noise
        )
        
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.model.fit(X_scaled, y)
        
        # Save model
        os.makedirs('models', exist_ok=True)
        with open('models/nutrition_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def generate_weekly_meal_plan(self, user_goals):
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        meal_types = ['breakfast', 'lunch', 'dinner']
        
        weekly_plan = {}
        
        for day in days:
            daily_plan = {}
            daily_calories = 0
            target_calories = user_goals['calories']
            
            for meal_type in meal_types:
                if meal_type == 'breakfast':
                    meal_calories = target_calories * 0.25
                elif meal_type == 'lunch':
                    meal_calories = target_calories * 0.35
                else:  # dinner
                    meal_calories = target_calories * 0.40
                
                meal_foods = self.recommend_meal_foods(
                    meal_type, 
                    meal_calories, 
                    user_goals.get('dietary_restrictions', [])
                )
                
                daily_plan[meal_type] = meal_foods
                daily_calories += sum(food['calories_per_100g'] for food in meal_foods)
            
            # Add healthy snacks if under calorie goal
            if daily_calories < target_calories * 0.9:
                snacks = self.recommend_meal_foods('snack', target_calories * 0.1, [])
                daily_plan['snacks'] = snacks
            
            weekly_plan[day] = daily_plan
        
        return weekly_plan
    
    def recommend_meal_foods(self, meal_type, target_calories, restrictions):
        # Filter foods by meal type and restrictions
        suitable_foods = self.food_database[
            (self.food_database['meal_type'].isin([meal_type, 'any']))
        ].copy()
        
        # Apply dietary restrictions
        if 'vegetarian' in restrictions:
            suitable_foods = suitable_foods[
                ~suitable_foods['category'].isin(['Protein']) |
                suitable_foods['food_name'].isin(['Tofu', 'Tempeh', 'Eggs'])
            ]
        
        if 'vegan' in restrictions:
            suitable_foods = suitable_foods[
                ~suitable_foods['category'].isin(['Protein', 'Dairy']) |
                suitable_foods['food_name'].isin(['Tofu', 'Tempeh'])
            ]
        
        # Select foods that fit calorie target
        if len(suitable_foods) > 0:
            # Use ML model to score foods
            features = ['calories_per_100g', 'protein', 'carbs', 'fat', 'fiber']
            X = suitable_foods[features]
            X_scaled = self.scaler.transform(X)
            scores = self.model.predict(X_scaled)
            
            suitable_foods['ml_score'] = scores
            
            # Select top-rated foods that fit calorie budget
            selected_foods = []
            remaining_calories = target_calories
            
            sorted_foods = suitable_foods.sort_values('ml_score', ascending=False)
            
            for _, food in sorted_foods.iterrows():
                if food['calories_per_100g'] <= remaining_calories and len(selected_foods) < 3:
                    selected_foods.append(food.to_dict())
                    remaining_calories -= food['calories_per_100g']
                    if remaining_calories <= 0:
                        break
            
            return selected_foods
        
        return []
    
    def get_daily_suggestions(self, goals):
        suggestions = {
            'breakfast': self.recommend_meal_foods('breakfast', goals['calories'] * 0.25, []),
            'lunch': self.recommend_meal_foods('lunch', goals['calories'] * 0.35, []),
            'dinner': self.recommend_meal_foods('dinner', goals['calories'] * 0.40, [])
        }
        return suggestions