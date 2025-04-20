from data import DataRepository
from naive_bayes import NaiveBayesClassifier
import numpy as np
import pandas as pd

# Load data
repo = DataRepository()
df = repo.load_data()

# Train, Test data
df_train = df[df['cat']=='training']
df_test = df[df['cat']=='testing']

# Load Model
model = NaiveBayesClassifier()

model.fit(df_train['content'].values, df_train['topic'].values)

pred = []
for content in df_test['content']:
    pred.append(model.predict_one(content))
    
df_test['pred'] = pred

acc = (df_test['topic'] == df_test['pred']).sum() / len(df_test)
        
print(f"Accuracy: {acc:.0%}")

