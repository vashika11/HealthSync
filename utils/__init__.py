### utils/__init__.py

"""
HealthSync Utilities Package

This package contains utility functions and helper classes for the HealthSync
nutrition tracking and meal planning application.

Modules:
    - data_processor: Data processing and nutrition calculation utilities
    - validators: Input validation and data sanitization functions

Functions:
    - quick_nutrition_calc: Quick nutrition calculation helper
    - format_nutrition_display: Format nutrition values for display
    - generate_meal_id: Generate unique meal identifiers

Classes:
    - DataProcessor: Main data processing class
    - NutritionCalculator: Advanced nutrition calculations
    - MealPlanGenerator: Meal plan generation utilities
"""

__version__ = "1.0.0"
__author__ = "HealthSync Development Team"
__email__ = "dev@healthsync.com"

# Import main utility functions and classes
from .data_processor import DataProcessor
from .validators import validate_user_input, sanitize_string

# Quick utility functions
import re
from typing import Dict, List, Union, Optional

def quick_nutrition_calc(foods: List[Dict], portions: List[float] = None) -> Dict[str, float]:
    """
    Quick nutrition calculation for a list of foods
    
    Args:
        foods: List of food dictionaries with nutrition info
        portions: List of portion sizes in grams (defaults to 100g each)
    
    Returns:
        Dict with total nutrition values
    """
    if portions is None:
        portions = [100.0] * len(foods)
    
    if len(foods) != len(portions):
        raise ValueError("Number of foods and portions must match")
    
    totals = {
        'calories': 0.0,
        'protein': 0.0,
        'carbs': 0.0,
        'fat': 0.0,
        'fiber': 0.0
    }
    
    for food, portion in zip(foods, portions):
        multiplier = portion / 100.0
        totals['calories'] += food.get('calories_per_100g', 0) * multiplier
        totals['protein'] += food.get('protein', 0) * multiplier
        totals['carbs'] += food.get('carbs', 0) * multiplier
        totals['fat'] += food.get('fat', 0) * multiplier
        totals['fiber'] += food.get('fiber', 0) * multiplier
    
    return totals

def format_nutrition_display(nutrition: Dict[str, float], precision: int = 1) -> Dict[str, str]:
    """
    Format nutrition values for display with appropriate units
    
    Args:
        nutrition: Dictionary with nutrition values
        precision: Decimal places for formatting
    
    Returns:
        Dict with formatted nutrition strings
    """
    formatted = {}
    
    # Format calories (no decimal for whole numbers)
    calories = nutrition.get('calories', 0)
    formatted['calories'] = f"{calories:.0f} cal" if calories == int(calories) else f"{calories:.{precision}f} cal"
    
    # Format macronutrients in grams
    for nutrient in ['protein', 'carbs', 'fat', 'fiber']:
        value = nutrition.get(nutrient, 0)
        formatted[nutrient] = f"{value:.{precision}f}g"
    
    return formatted

def generate_meal_id(user_id: str, meal_type: str, date: str = None) -> str:
    """
    Generate a unique meal identifier
    
    Args:
        user_id: User identifier
        meal_type: Type of meal (breakfast, lunch, dinner, snack)
        date: Date string (defaults to today)
    
    Returns:
        Unique meal ID string
    """
    import datetime
    import hashlib
    
    if date is None:
        date = datetime.date.today().isoformat()
    
    # Create a hash from user_id, meal_type, and date
    hash_input = f"{user_id}_{meal_type}_{date}".encode('utf-8')
    hash_object = hashlib.md5(hash_input)
    
    return f"meal_{hash_object.hexdigest()[:8]}"

def calculate_bmi(weight_kg: float, height_cm: float) -> Dict[str, Union[float, str]]:
    """
    Calculate BMI and return with category
    
    Args:
        weight_kg: Weight in kilograms
        height_cm: Height in centimeters
    
    Returns:
        Dict with BMI value and category
    """
    if weight_kg <= 0 or height_cm <= 0:
        raise ValueError("Weight and height must be positive values")
    
    height_m = height_cm / 100.0
    bmi = weight_kg / (height_m ** 2)
    
    # Determine BMI category
    if bmi < 18.5:
        category = "Underweight"
        color = "blue"
    elif bmi < 25:
        category = "Normal weight"
        color = "green"
    elif bmi < 30:
        category = "Overweight"
        color = "orange"
    else:
        category = "Obese"
        color = "red"
    
    return {
        'bmi': round(bmi, 1),
        'category': category,
        'color': color
    }

def estimate_daily_calories(weight_kg: float, height_cm: float, age: int, 
                          gender: str, activity_level: str = 'moderate') -> int:
    """
    Estimate daily calorie needs using Mifflin-St Jeor Equation
    
    Args:
        weight_kg: Weight in kilograms
        height_cm: Height in centimeters
        age: Age in years
        gender: 'male' or 'female'
        activity_level: Activity level (sedentary, light, moderate, heavy, extra)
    
    Returns:
        Estimated daily calories
    """
    # Calculate Basal Metabolic Rate (BMR)
    if gender.lower() == 'male':
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    
    # Activity multipliers
    activity_multipliers = {
        'sedentary': 1.2,
        'light': 1.375,
        'moderate': 1.55,
        'heavy': 1.725,
        'extra': 1.9
    }
    
    multiplier = activity_multipliers.get(activity_level.lower(), 1.55)
    total_calories = bmr * multiplier
    
    return int(round(total_calories))

def parse_food_input(food_string: str) -> List[str]:
    """
    Parse a string of food items into a clean list
    
    Args:
        food_string: Comma or semicolon separated food items
    
    Returns:
        List of clean food names
    """
    # Split by comma or semicolon
    foods = re.split(r'[,;]', food_string)
    
    # Clean each food item
    cleaned_foods = []
    for food in foods:
        food = food.strip()
        if food:  # Skip empty strings
            # Remove extra whitespace and capitalize properly
            food = re.sub(r'\s+', ' ', food)
            food = food.title()
            cleaned_foods.append(food)
    
    return cleaned_foods

def get_nutrition_color(current: float, target: float) -> str:
    """
    Get color code based on nutrition progress
    
    Args:
        current: Current nutrition value
        target: Target nutrition value
    
    Returns:
        Color class name for CSS styling
    """
    if target == 0:
        return 'gray'
    
    percentage = (current / target) * 100
    
    if percentage < 50:
        return 'red'
    elif percentage < 80:
        return 'orange'
    elif percentage <= 110:
        return 'green'
    else:
        return 'blue'  # Over target

def create_nutrition_summary(meals: List[Dict]) -> Dict[str, float]:
    """
    Create a nutrition summary from a list of meals
    
    Args:
        meals: List of meal dictionaries with nutrition info
    
    Returns:
        Dict with total nutrition summary
    """
    summary = {
        'total_calories': 0.0,
        'total_protein': 0.0,
        'total_carbs': 0.0,
        'total_fat': 0.0,
        'total_fiber': 0.0,
        'meal_count': len(meals)
    }
    
    for meal in meals:
        nutrition = meal.get('nutrition', {})
        summary['total_calories'] += nutrition.get('calories', 0)
        summary['total_protein'] += nutrition.get('protein', 0)
        summary['total_carbs'] += nutrition.get('carbs', 0)
        summary['total_fat'] += nutrition.get('fat', 0)
        summary['total_fiber'] += nutrition.get('fiber', 0)
    
    # Calculate averages
    if summary['meal_count'] > 0:
        summary['avg_calories'] = summary['total_calories'] / summary['meal_count']
        summary['avg_protein'] = summary['total_protein'] / summary['meal_count']
    else:
        summary['avg_calories'] = 0.0
        summary['avg_protein'] = 0.0
    
    return summary

def validate_nutrition_goals(goals: Dict[str, Union[int, float]]) -> Dict[str, str]:
    """
    Validate nutrition goals and return any warnings
    
    Args:
        goals: Dictionary with nutrition goals
    
    Returns:
        Dict with validation warnings (empty if all valid)
    """
    warnings = {}
    
    calories = goals.get('calories', 0)
    protein = goals.get('protein', 0)
    carbs = goals.get('carbs', 0)
    fat = goals.get('fat', 0)
    
    # Check calorie range
    if calories < 1200:
        warnings['calories'] = "Very low calorie intake may not be sustainable"
    elif calories > 4000:
        warnings['calories'] = "Very high calorie intake - consult a nutritionist"
    
    # Check protein percentage (should be 10-35% of calories)
    protein_calories = protein * 4
    protein_percentage = (protein_calories / calories) * 100 if calories > 0 else 0
    
    if protein_percentage < 10:
        warnings['protein'] = "Protein intake may be too low (aim for 10-35% of calories)"
    elif protein_percentage > 35:
        warnings['protein'] = "Very high protein intake - ensure adequate hydration"
    
    # Check carb percentage (should be 45-65% of calories)
    carb_calories = carbs * 4
    carb_percentage = (carb_calories / calories) * 100 if calories > 0 else 0
    
    if carb_percentage < 20:
        warnings['carbs'] = "Very low carb intake - monitor energy levels"
    elif carb_percentage > 70:
        warnings['carbs'] = "High carb intake - ensure balance with protein and fats"
    
    # Check fat percentage (should be 20-35% of calories)
    fat_calories = fat * 9
    fat_percentage = (fat_calories / calories) * 100 if calories > 0 else 0
    
    if fat_percentage < 15:
        warnings['fat'] = "Low fat intake may affect vitamin absorption"
    elif fat_percentage > 40:
        warnings['fat'] = "High fat intake - focus on healthy fats"
    
    return warnings

# Export all utility functions
__all__ = [
    'DataProcessor',
    'validate_user_input',
    'sanitize_string',
    'quick_nutrition_calc',
    'format_nutrition_display',
    'generate_meal_id',
    'calculate_bmi',
    'estimate_daily_calories',
    'parse_food_input',
    'get_nutrition_color',
    'create_nutrition_summary',
    'validate_nutrition_goals'
]

# Package metadata
SUPPORTED_PYTHON_VERSIONS = ["3.7", "3.8", "3.9", "3.10", "3.11"]
REQUIRED_PACKAGES = ["pandas", "numpy", "flask"]

# Utility constants
MEAL_TYPES = ['breakfast', 'lunch', 'dinner', 'snack']
NUTRIENTS = ['calories', 'protein', 'carbs', 'fat', 'fiber']
FOOD_CATEGORIES = [
    'Fruit', 'Vegetable', 'Protein', 'Grain', 'Dairy', 
    'Nuts', 'Seeds', 'Legume', 'Healthy Fat', 'Beverage'
]

# Default nutrition targets (per day)
DEFAULT_NUTRITION_TARGETS = {
    'calories': 2000,
    'protein': 150,
    'carbs': 250,
    'fat': 70,
    'fiber': 25
}

# Activity level descriptions
ACTIVITY_LEVELS = {
    'sedentary': 'Little or no exercise',
    'light': 'Light exercise/sports 1-3 days/week',
    'moderate': 'Moderate exercise/sports 3-5 days/week',
    'heavy': 'Heavy exercise/sports 6-7 days a week',
    'extra': 'Very heavy exercise/sports & physical job'
}

def get_package_info() -> Dict[str, Union[str, List[str]]]:
    """
    Get package information
    
    Returns:
        Dict with package metadata
    """
    return {
        'name': 'HealthSync Utils',
        'version': __version__,
        'author': __author__,
        'email': __email__,
        'supported_python': SUPPORTED_PYTHON_VERSIONS,
        'required_packages': REQUIRED_PACKAGES,
        'meal_types': MEAL_TYPES,
        'nutrients': NUTRIENTS,
        'food_categories': FOOD_CATEGORIES
    }

# Initialize logging for the utils package
import logging

# Create a logger for the utils package
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Only add handler if none exists to prevent duplicate logs
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Log package initialization
logger.info(f"HealthSync Utils v{__version__} initialized")