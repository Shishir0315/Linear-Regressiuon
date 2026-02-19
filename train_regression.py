import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def load_and_clean_data(file_path):
    # Try reading locally, if it fails due to the malformed line, we handle it
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        # Manual cleaning if needed
        return None
    
    # Drop NAs
    df = df.dropna()
    
    # Ensure numeric
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df = df.dropna()
    
    return df

def build_model():
    model = tf.keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(1,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss='mse',
        metrics=['mae']
    )
    return model

def main():
    train_path = r'c:\Users\student\Desktop\regression\train.csv'
    test_path = r'c:\Users\student\Desktop\regression\test.csv'
    
    print("Loading data...")
    train_df = load_and_clean_data(train_path)
    test_df = load_and_clean_data(test_path)
    
    if train_df is None or test_df is None:
        print("Data loading failed.")
        return

    # Use less data as requested (e.g., first 500 rows)
    train_subset = train_df.head(500)
    X_train = train_subset[['x']].values
    y_train = train_subset['y'].values
    
    X_test = test_df[['x']].values
    y_test = test_df['y'].values
    
    print(f"Training on {len(X_train)} samples.")
    
    model = build_model()
    
    # Use less epochs as requested
    epochs = 20
    print(f"Training for {epochs} epochs...")
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=32,
        verbose=1
    )
    
    print("\nEvaluating model...")
    test_results = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test MSE: {test_results[0]:.4f}")
    print(f"Test MAE: {test_results[1]:.4f}")
    
    # Save model
    model_save_path = r'c:\Users\student\Desktop\regression\regression_model.keras'
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Make a prediction
    sample_x = np.array([[50.0]])
    prediction = model.predict(sample_x)
    print(f"Prediction for x=50: {prediction[0][0]:.4f}")
    
if __name__ == "__main__":
    main()
