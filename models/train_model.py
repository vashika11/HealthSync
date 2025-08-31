#!/usr/bin/env python3
"""
HealthSync Model Training Script

This script trains the machine learning models used for nutrition recommendations.
Run this script to create or update the ML models.

Usage:
    python models/train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class NutritionModelTrainer:
    def __init__(self, data_path='data/nutrition.csv'):
        """Initialize the model trainer"""
        self.data_path = data_path
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = ['calories_per_100g', 'protein', 'carbs', 'fat', 'fiber']
        self.model_save_path = 'models/saved/'
        
        # Create save directory if it doesn't exist
        os.makedirs(self.model_save_path, exist_ok=True)
        
        print("HealthSync Nutrition Model Trainer Initialized")
        print(f"Data path: {self.data_path}")
        print(f"Model save path: {self.model_save_path}")
    
    def load_and_prepare_data(self):
        """Load and prepare the nutrition data for training"""
        try:
            print("\n" + "="*50)
            print("LOADING AND PREPARING DATA")
            print("="*50)
            
            # Load the data
            self.df = pd.read_csv(self.data_path)
            print(f"Loaded {len(self.df)} food items from {self.data_path}")
            
            # Display basic info about the dataset
            print(f"\nDataset shape: {self.df.shape}")
            print(f"Columns: {list(self.df.columns)}")
            
            # Check for missing values
            missing_values = self.df.isnull().sum()
            if missing_values.any():
                print(f"\nMissing values found:")
                for col, count in missing_values.items():
                    if count > 0:
                        print(f"  {col}: {count}")
            else:
                print("\nNo missing values found ✓")
            
            # Display basic statistics
            print(f"\nNutrition Data Summary:")
            print(self.df[self.feature_columns].describe().round(2))
            
            # Display category distribution
            if 'category' in self.df.columns:
                print(f"\nFood Categories:")
                category_counts = self.df['category'].value_counts()
                for category, count in category_counts.items():
                    print(f"  {category}: {count} items")
            
            return True
            
        except FileNotFoundError:
            print(f"Error: Could not find data file at {self.data_path}")
            print("Please ensure the nutrition.csv file exists in the data/ directory")
            return False
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
    
    def create_target_variables(self):
        """Create target variables for training"""
        print("\n" + "="*50)
        print("CREATING TARGET VARIABLES")
        print("="*50)
        
        # Nutritional Quality Score (primary target)
        # Higher protein and fiber are good, lower calories per unit nutrition is better
        self.df['nutrition_density'] = (
            self.df['protein'] * 0.3 +           # Protein importance
            self.df['fiber'] * 0.25 +            # Fiber importance
            (100 / (self.df['calories_per_100g'] + 1)) * 0.15 +  # Caloric efficiency
            np.random.normal(0, 0.1, len(self.df))  # Add some noise for variety
        )
        
        # Meal suitability scores for different meal types
        breakfast_foods = ['Fruit', 'Dairy', 'Grain']
        lunch_foods = ['Protein', 'Vegetable', 'Grain', 'Legume']
        dinner_foods = ['Protein', 'Vegetable', 'Healthy Fat']
        
        self.df['breakfast_score'] = self.df['category'].apply(
            lambda x: 0.8 + np.random.normal(0, 0.1) if x in breakfast_foods else 0.2 + np.random.normal(0, 0.1)
        )
        
        self.df['lunch_score'] = self.df['category'].apply(
            lambda x: 0.8 + np.random.normal(0, 0.1) if x in lunch_foods else 0.3 + np.random.normal(0, 0.1)
        )
        
        self.df['dinner_score'] = self.df['category'].apply(
            lambda x: 0.8 + np.random.normal(0, 0.1) if x in dinner_foods else 0.3 + np.random.normal(0, 0.1)
        )
        
        # Weight loss friendliness (lower calories, higher protein/fiber)
        self.df['weight_loss_score'] = (
            (200 - self.df['calories_per_100g']) / 200 * 0.4 +  # Lower calories better
            self.df['protein'] / 50 * 0.3 +                     # Higher protein better
            self.df['fiber'] / 20 * 0.3                         # Higher fiber better
        ).clip(0, 1)
        
        print("Created target variables:")
        print(f"  - nutrition_density: {self.df['nutrition_density'].min():.2f} to {self.df['nutrition_density'].max():.2f}")
        print(f"  - breakfast_score: {self.df['breakfast_score'].min():.2f} to {self.df['breakfast_score'].max():.2f}")
        print(f"  - lunch_score: {self.df['lunch_score'].min():.2f} to {self.df['lunch_score'].max():.2f}")
        print(f"  - dinner_score: {self.df['dinner_score'].min():.2f} to {self.df['dinner_score'].max():.2f}")
        print(f"  - weight_loss_score: {self.df['weight_loss_score'].min():.2f} to {self.df['weight_loss_score'].max():.2f}")
    
    def prepare_features(self):
        """Prepare features for training"""
        print("\n" + "="*50)
        print("PREPARING FEATURES")
        print("="*50)
        
        # Basic nutrition features
        X_basic = self.df[self.feature_columns].copy()
        
        # Create additional engineered features
        X_basic['protein_to_calorie_ratio'] = X_basic['protein'] / (X_basic['calories_per_100g'] + 1)
        X_basic['fiber_to_calorie_ratio'] = X_basic['fiber'] / (X_basic['calories_per_100g'] + 1)
        X_basic['protein_carb_ratio'] = X_basic['protein'] / (X_basic['carbs'] + 1)
        X_basic['total_macros'] = X_basic['protein'] + X_basic['carbs'] + X_basic['fat']
        
        # Log transform for skewed features (add 1 to avoid log(0))
        X_basic['log_calories'] = np.log(X_basic['calories_per_100g'] + 1)
        X_basic['sqrt_fiber'] = np.sqrt(X_basic['fiber'] + 1)
        
        # Category encoding
        if 'category' in self.df.columns:
            category_encoded = pd.get_dummies(self.df['category'], prefix='cat')
            X_basic = pd.concat([X_basic, category_encoded], axis=1)
        
        print(f"Feature matrix shape: {X_basic.shape}")
        print(f"Features created: {list(X_basic.columns)}")
        
        return X_basic
    
    def train_models(self, X, target_name='nutrition_density'):
        """Train multiple models and select the best one"""
        print(f"\n" + "="*50)
        print(f"TRAINING MODELS FOR {target_name.upper()}")
        print("="*50)
        
        y = self.df[target_name]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models to try
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=6),
            'Random Forest (Tuned)': RandomForestRegressor(
                n_estimators=200, max_depth=15, min_samples_split=5, 
                min_samples_leaf=2, random_state=42
            )
        }
        
        best_model = None
        best_score = float('-inf')
        best_name = ""
        
        print("\nTraining and evaluating models...")
        
        for name, model in models.items():
            print(f"\n--- {name} ---")
            
            # Train the model
            if 'Random Forest' in name or 'Gradient Boosting' in name:
                model.fit(X_train, y_train)  # Tree-based models don't need scaling
                y_pred = model.predict(X_test)
            else:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"R²: {r2:.4f}")
            
            # Cross-validation
            if 'Random Forest' in name or 'Gradient Boosting' in name:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            else:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            
            print(f"Cross-val R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            # Select best model based on R²
            if r2 > best_score:
                best_score = r2
                best_model = model
                best_name = name
        
        print(f"\n🏆 Best model: {best_name} (R² = {best_score:.4f})")
        
        # Feature importance for tree-based models
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 10 Feature Importances for {target_name}:")
            for idx, row in feature_importance.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return best_model, best_name, best_score
    
    def save_models(self, models_dict):
        """Save trained models to disk"""
        print("\n" + "="*50)
        print("SAVING MODELS")
        print("="*50)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save individual models
        for target_name, (model, model_name, score) in models_dict.items():
            filename = f'{self.model_save_path}{target_name}_model.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
            print(f"✓ Saved {target_name} model ({model_name}, R²={score:.4f}) to {filename}")
        
        # Save the scaler
        scaler_filename = f'{self.model_save_path}scaler.pkl'
        with open(scaler_filename, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✓ Saved scaler to {scaler_filename}")
        
        # Save model metadata
        metadata = {
            'timestamp': timestamp,
            'data_path': self.data_path,
            'feature_columns': self.feature_columns,
            'models': {name: {'type': info[1], 'score': info[2]} for name, info in models_dict.items()},
            'total_samples': len(self.df)
        }
        
        metadata_filename = f'{self.model_save_path}model_metadata.pkl'
        with open(metadata_filename, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"✓ Saved metadata to {metadata_filename}")
        
        print(f"\n🎉 All models saved successfully!")
        print(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def run_full_training(self):
        """Run the complete training pipeline"""
        print("🚀 Starting HealthSync Model Training Pipeline")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Step 1: Load and prepare data
        if not self.load_and_prepare_data():
            print("❌ Training failed: Could not load data")
            return False
        
        # Step 2: Create target variables
        self.create_target_variables()
        
        # Step 3: Prepare features
        X = self.prepare_features()
        
        # Step 4: Train models for different targets
        targets = ['nutrition_density', 'breakfast_score', 'lunch_score', 'dinner_score', 'weight_loss_score']
        trained_models = {}
        
        for target in targets:
            model, name, score = self.train_models(X, target)
            trained_models[target] = (model, name, score)
        
        # Step 5: Save all models
        self.save_models(trained_models)
        
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        
        for target, (model, name, score) in trained_models.items():
            print(f"{target:20} | {name:20} | R² = {score:.4f}")
        
        print(f"\n✅ Training completed successfully!")
        print(f"Models saved to: {os.path.abspath(self.model_save_path)}")
        
        return True

def main():
    """Main function to run model training"""
    try:
        trainer = NutritionModelTrainer()
        success = trainer.run_full_training()
        
        if success:
            print(f"\n🎯 Next steps:")
            print(f"   1. Start the Flask application: python app.py")
            print(f"   2. The models will be automatically loaded")
            print(f"   3. Test the recommendation system")
        else:
            print(f"\n❌ Training failed. Please check the error messages above.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n⏹️ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()