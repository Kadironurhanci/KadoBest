This project implements a Hybrid Machine Learning & Deep Learning architecture for real-time time-series forecasting.

The system integrates:

LSTM-based Deep Learning model (TensorFlow – primary framework)

Random Forest Regressor

Support Vector Regressor (SVR)

It features a real-time adaptive prediction pipeline with SQL database integration and error-feedback correction mechanism.

Architecture

Data is continuously fetched from Microsoft SQL Server.

Data preprocessing includes:

Rolling window segmentation

MinMaxScaler normalization

Models generate predictions:

LSTM (Deep Learning)

Random Forest

SVR

Predictions are compared with real incoming values.

Error values are stored and used for adaptive correction.

System updates predictions using mean error compensation.

This enables semi-online learning behavior.

Key Features

Hybrid ML/DL Ensemble Architecture

Real-Time Prediction Pipeline

SQL Server Integration

Error-Based Adaptive Correction Mechanism

Early Stopping to Prevent Overfitting

Hyperparameter Tuning Experiments

Cross-Validation & Model Performance Comparison

Lightweight Monitoring Interface

Technologies Used

Python

TensorFlow

PyTorch (experimental)

Scikit-learn

SQL Server

Pandas

NumPy

Matplotlib

Model Training Details

12,000+ sequential data points

Rolling Window technique for time-series transformation

Mean Squared Error (MSE) loss function

Early Stopping applied

Hyperparameter tuning (epochs, batch size, window size)

Feature scaling using MinMaxScaler

How It Works

The system continuously:

Fetches latest data from database

Applies preprocessing

Generates predictions

Compares predictions with actual values

Calculates prediction error

Adjusts next predictions using mean error compensation

This creates a feedback-driven adaptive prediction system.

Future Improvements

Full online learning implementation

Deployment with REST API

Model containerization (Docker)

CI/CD integration

Author

Kadir Onur Hancı
Artificial Intelligence & Machine Learning Enthusiast

GitHub: https://github.com/Kadironurhanci
