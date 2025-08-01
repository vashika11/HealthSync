# model/train_model.py
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle

df = pd.read_csv('data/nutrition.csv')
features = df[['Calories', 'Protein', 'Fat', 'Carbs', 'Fiber']]

model = NearestNeighbors(n_neighbors=3)
model.fit(features)

with open('model/diet_model.pkl', 'wb') as f:
    pickle.dump(model, f)