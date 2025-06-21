import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Simulate Sensor Data (Same as before)
def simulate_data(num_samples=1000):
    np.random.seed(42)
    cutting_speed = np.random.uniform(100, 300, num_samples)
    feed_rate = np.random.uniform(0.1, 0.3, num_samples)
    depth_of_cut = np.random.uniform(1, 3, num_samples)
    spindle_speed = np.random.uniform(2000, 5000, num_samples)
    force_x = 10 + 0.5 * cutting_speed + 2 * depth_of_cut + np.random.normal(0, 2, num_samples)
    force_y = 5 + 0.3 * feed_rate * spindle_speed + np.random.normal(0, 1.5, num_samples)
    vibration = 0.1 * spindle_speed + 0.2 * depth_of_cut * force_x + np.random.normal(0, 0.5, num_samples)
    tool_wear = 0.01 * cutting_speed + 0.05 * feed_rate * spindle_speed + 0.1 * depth_of_cut + 0.001 * force_x * force_y + np.random.normal(0, 0.2, num_samples)
    tool_wear = np.clip(tool_wear, 0, 5)

    data = pd.DataFrame({
        'CuttingSpeed': cutting_speed,
        'FeedRate': feed_rate,
        'DepthOfCut': depth_of_cut,
        'SpindleSpeed': spindle_speed,
        'ForceX': force_x,
        'ForceY': force_y,
        'Vibration': vibration,
        'ToolWear': tool_wear
    })
    return data

# 2. Data Preprocessing (Same as before)
def preprocess_data(data):
    numerical_features = ['CuttingSpeed', 'FeedRate', 'DepthOfCut', 'SpindleSpeed', 'ForceX', 'ForceY', 'Vibration']
    scaler = StandardScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])
    return data

# 3. Feature Engineering (Same as before)
def feature_engineering(data):
    data['CuttingSpeed_x_FeedRate'] = data['CuttingSpeed'] * data['FeedRate']
    return data

# 4. Train a Machine Learning Model (Same as before, but returns the test data too)
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test, y_train  # Return y_train as well

# 5. Evaluate the Model (Modified for Streamlit)
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2, y_pred

# 6. Streamlit App
def main():
    st.title("Tool Wear Prediction in CNC Machining")

    # Dataset Upload
    st.sidebar.header("Data Input")
    data_source = st.sidebar.radio("Select Data Source:", ("Simulated Data", "Upload CSV"))

    if data_source == "Simulated Data":
        data = simulate_data()
        st.write("Using Simulated Data")
    else:
        uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.write("Data uploaded successfully!")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
                return  # Stop if there's an error

        else:
            st.info("Please upload a CSV file or select 'Simulated Data'.")
            return  # Stop if no data is provided

    # Display Data
    st.header("Data Preview")
    st.dataframe(data.head())

    # Check if 'ToolWear' column exists
    if 'ToolWear' not in data.columns:
        st.error("The uploaded CSV file must contain a 'ToolWear' column.")
        return

    # Preprocessing and Feature Engineering
    data = preprocess_data(data.copy())  # Use a copy to avoid modifying the original
    data = feature_engineering(data.copy())  # Use a copy

    # Prepare data for model training
    try:
        X = data.drop('ToolWear', axis=1)
        y = data['ToolWear']
    except KeyError as e:
        st.error(f"Error: Missing column in data.  Make sure your data has the correct columns. {e}")
        return

    # Train the model
    model, X_test, y_test, y_train = train_model(X, y)

    # Evaluate the model
    mse, r2, y_pred = evaluate_model(model, X_test, y_test)

    # Display Results
    st.header("Model Performance")
    st.write(f"Mean Squared Error: {mse:.4f}")
    st.write(f"R-squared: {r2:.4f}")

    # Plot predicted vs. actual values
    st.header("Prediction Visualization")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3) # Add a y=x line
    ax.set_xlabel("Actual Tool Wear")
    ax.set_ylabel("Predicted Tool Wear")
    ax.set_title("Actual vs. Predicted Tool Wear")
    st.pyplot(fig)

if __name__ == "__main__":
    main()