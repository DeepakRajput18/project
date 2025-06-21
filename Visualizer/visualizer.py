import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st  # For optional Streamlit integration

# Path to the predictions file
PREDICTIONS_FILE = "tool_wear_predictions.csv"

def load_predictions(file_path):
    """Loads predictions from a CSV file."""
    try:
        predictions_df = pd.read_csv(file_path)
        return predictions_df
    except FileNotFoundError:
        print(f"Error: Predictions file not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading predictions: {e}")
        return None

def plot_actual_vs_predicted(predictions_df, use_streamlit=False):
    """Plots actual vs. predicted values."""
    if predictions_df is None:
        return

    plt.figure(figsize=(8, 6))
    plt.scatter(predictions_df['Actual'], predictions_df['Predicted'], alpha=0.5)
    plt.plot([predictions_df['Actual'].min(), predictions_df['Actual'].max()],
             [predictions_df['Actual'].min(), predictions_df['Actual'].max()],
             'k--', lw=2, label='Ideal')  # Add ideal line
    plt.xlabel("Actual Tool Wear")
    plt.ylabel("Predicted Tool Wear")
    plt.title("Actual vs. Predicted Tool Wear")
    plt.legend()
    plt.grid(True)

    if use_streamlit:
        st.pyplot(plt)  # Display in Streamlit
    else:
        plt.show()  # Display in a separate window

def plot_residuals(predictions_df, use_streamlit=False):
    """Plots the residuals (errors)."""
    if predictions_df is None:
        return

    residuals = predictions_df['Actual'] - predictions_df['Predicted']

    plt.figure(figsize=(8, 6))
    plt.scatter(predictions_df['Actual'], residuals, alpha=0.5)
    plt.hlines(y=0, xmin=predictions_df['Actual'].min(), xmax=predictions_df['Actual'].max(),
               color='red', linestyle='--', label='Zero Error')  # Add zero line
    plt.xlabel("Actual Tool Wear")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title("Residual Plot")
    plt.legend()
    plt.grid(True)

    if use_streamlit:
        st.pyplot(plt)  # Display in Streamlit
    else:
        plt.show()  # Display in a separate window

def plot_residual_histogram(predictions_df, use_streamlit=False):
    """Plots a histogram of the residuals."""
    if predictions_df is None:
        return

    residuals = predictions_df['Actual'] - predictions_df['Predicted']

    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True)  # Use seaborn for better histogram
    plt.xlabel("Residuals (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Residuals")
    plt.grid(True)

    if use_streamlit:
        st.pyplot(plt)  # Display in Streamlit
    else:
        plt.show()  # Display in a separate window

def main():
    """Main function to load predictions and generate visualizations."""
    predictions_df = load_predictions(PREDICTIONS_FILE)

    if predictions_df is not None:
        print("Generating visualizations...")
        plot_actual_vs_predicted(predictions_df)
        plot_residuals(predictions_df)
        plot_residual_histogram(predictions_df)
        print("Visualizations generated.  Check the plot windows.")
    else:
        print("No predictions data to visualize.")

# Optional Streamlit integration
def streamlit_visualizer():
    """Streamlit app to display visualizations."""
    st.title("Tool Wear Prediction Visualizations")
    predictions_df = load_predictions(PREDICTIONS_FILE)

    if predictions_df is not None:
        st.header("Actual vs. Predicted")
        plot_actual_vs_predicted(predictions_df, use_streamlit=True)

        st.header("Residual Plot")
        plot_residuals(predictions_df, use_streamlit=True)

        st.header("Residual Histogram")
        plot_residual_histogram(predictions_df, use_streamlit=True)
    else:
        st.error("No predictions data to visualize.  Make sure the scheduler has run and generated the predictions file.")

if __name__ == "__main__":
    # Choose either the command-line version or the Streamlit version
    # To run the command-line version: python visualizer.py
    # To run the Streamlit version: streamlit run visualizer.py -- --streamlit
    import sys
    if "--streamlit" in sys.argv:
        streamlit_visualizer()
    else:
        main()