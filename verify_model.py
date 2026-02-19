import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

def main():
    # Load data
    test_path = r'c:\Users\student\Desktop\regression\test.csv'
    df_test = pd.read_csv(test_path).dropna()
    df_test['x'] = pd.to_numeric(df_test['x'], errors='coerce')
    df_test['y'] = pd.to_numeric(df_test['y'], errors='coerce')
    df_test = df_test.dropna()
    
    X_test = df_test[['x']].values
    y_test = df_test['y'].values
    
    # Load model
    model_path = r'c:\Users\student\Desktop\regression\regression_model.keras'
    if not os.path.exists(model_path):
        print("Model file not found!")
        return
        
    model = tf.keras.models.load_model(model_path)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Visualize
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Actual Data')
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line (Model)')
    plt.title('Regression Model Performance (Test Set)', fontsize=14)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save visualization
    plot_path = r'c:\Users\student\Desktop\regression\results_plot.png'
    plt.savefig(plot_path, dpi=300)
    print(f"Visualization saved to {plot_path}")
    
    # Show some sample predictions
    print("\nSample Predictions:")
    for i in range(5):
        print(f"X: {X_test[i][0]:.2f} | Actual Y: {y_test[i]:.2f} | Predicted Y: {y_pred[i][0]:.2f}")

if __name__ == "__main__":
    main()
