#!/bin/bash
# Build script for Render deployment

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Training model if needed..."
python train_model.py

echo "Build completed successfully!"