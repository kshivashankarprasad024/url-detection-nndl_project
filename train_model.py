#!/usr/bin/env python3
"""
Simple script to train and save the phishing detection model.
"""
import pandas as pd
import pickle
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

def train_model():
    # Create pickle directory if it doesn't exist
    pickle_dir = "pickle"
    if not os.path.exists(pickle_dir):
        os.makedirs(pickle_dir)
    
    # Check if model already exists
    model_path = os.path.join(pickle_dir, "model.pkl")
    if os.path.exists(model_path):
        print("Model already exists. Skipping training.")
        return
    
    # Load the dataset
    print("Loading dataset...")
    if not os.path.exists("phishing.csv"):
        print("Error: phishing.csv not found!")
        return
    
    data = pd.read_csv("phishing.csv")
    
    # Prepare the data - exclude Index column and class column
    X = data.drop(['Index', 'class'], axis=1)  # Features (exclude Index)
    y = data['class']  # Target variable
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    print("Training Gradient Boosting Classifier...")
    gbc = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gbc.fit(X_train, y_train)
    
    # Test accuracy
    accuracy = gbc.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.4f}")
    
    # Save the model
    print(f"Saving model to {model_path}...")
    with open(model_path, "wb") as f:
        pickle.dump(gbc, f)
    
    print("Model training completed successfully!")

if __name__ == "__main__":
    train_model()