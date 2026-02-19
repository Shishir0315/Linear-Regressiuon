---
title: Linear Regression
emoji: 📈
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
---

# Regression Pro: Intelligent Predictor

This project implements an end-to-end regression model trained on a custom dataset. It includes a simple but high-performance deep learning model and a premium web deployment interface.

## 🚀 Quick Start (Windows)

1.  **Clone/Extract** the project folder.
2.  **Run Setup**: Double-click `setup.bat` to install all required dependencies.
3.  **Run Application**: Open a terminal in this folder and run:
    ```bash
    py app.py
    ```
4.  **Access UI**: Open your browser and navigate to `http://127.0.0.1:5000`.

## 📦 Project Structure

*   `app.py`: Flask backend for serving the model.
*   `regression_model.keras`: The trained neural network (TensorFlow).
*   `templates/index.html`: Stunning, modern web interface.
*   `requirements.txt`: Python package dependencies.
*   `setup.bat`: One-click setup script for Windows.
*   `train_regression.py`: Script used to build and train the model.
*   `verify_model.py`: Script for local verification and plotting.

## 🧠 Model Details

*   **Architecture**: Sequential Feed-Forward Neural Network.
*   **Layers**: Dense(64) -> ReLU -> Dense(64) -> ReLU -> Dense(1).
*   **Optimizer**: Adam (lr=0.01).
*   **Loss**: Mean Squared Error (MSE).
*   **Training**: Optimized with subset data and limited epochs for efficiency.

## 🎨 UI Features

*   **Glassmorphism Design**: High-end translucent elements.
*   **Dynamic Background**: Animated radial gradients and floating blobs.
*   **Responsive**: Works on desktop and mobile devices.
*   **Real-time Predictions**: Instant feedback via AJAX calls to the Flask API.
