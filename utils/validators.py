def validate_user_input(user_data):
    """Validate user input for nutrition goals and meal planning"""
    
    # Required fields
    required_numeric = ['calories', 'protein']
    
    for field in required_numeric:
        if field not in user_data:
            return False
        
        try:
            value = float(user_data[field])
            if value <= 0:
                return False
        except (ValueError, TypeError):
            return False
    
    # Validate ranges
    calories = float(user_data.get('calories', 0))
    if calories < 800 or calories > 5000:
        return False
    
    protein = float(user_data.get('protein', 0))
    if protein < 20 or protein > 400:
        return False
    
    # Optional fields validation
    if 'carbs' in user_data:
        carbs = float(user_data['carbs'])
        if carbs < 0 or carbs > 800:
            return False
    
    if 'fat' in user_data:
        fat = float(user_data['fat'])
        if fat < 0 or fat > 300:
            return False
    
    return True

def sanitize_string(input_string, max_length=100):
    """Sanitize string inputs"""
    if not isinstance(input_string, str):
        return ""
    
    # Remove potentially harmful characters
    cleaned = input_string.strip()[:max_length]
    
    # Basic HTML escape
    cleaned = cleaned.replace('<', '&lt;').replace('>', '&gt;')
    cleaned = cleaned.replace('"', '&quot;').replace("'", '&#x27;')
    
    return cleaned