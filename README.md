# Infosys-springboard-internship-
#  Student Performance Prediction Dashboard (AI-Based)

An AI-powered web application that predicts student academic performance using Machine Learning and provides intelligent insights through an interactive dashboard.

---

## Project Overview

This project is a **Streamlit-based web dashboard** designed to:
- Predict student marks using an **AI model (Random Forest Regressor)**
- Analyze student behavior patterns
- Provide personalized recommendations
- Support both **single and bulk predictions**
- Visualize performance trends using interactive charts

The application includes a **secure login system** and works as a **24/7 web application**.

---

## Features

### Authentication
- User Login
- User Signup
- Forgot Password
- Credentials stored securely in SQLite database

###  AI & Machine Learning
- **AI-based Marks Prediction** using Random Forest Regressor
- Uses multiple input features:
  - Study Hours
  - Work Hours
  - Play Hours
  - Sleep Hours
- Model accuracy displayed using R² score

###  Predictions
- **Single Student Prediction**
- **Bulk Prediction** via CSV / Excel upload
- Downloadable prediction results

###  Intelligent Insights
- Automatic performance recommendations based on predicted marks
- AI-style feedback for improvement

###  Analytics & Visualization
- Cluster-based student analysis (K-Means)
- Scatter plot: Study Hours vs Marks
- Marks distribution histogram

---

## Technologies Used

- **Python**
- **Streamlit** – Web application framework
- **Pandas & NumPy** – Data processing
- **Scikit-learn**
  - RandomForestRegressor
  - KMeans
- **Plotly** – Interactive visualizations
- **SQLite** – Database for users & predictions

---

##  Project Structure