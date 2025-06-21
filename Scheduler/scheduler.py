import schedule
import time
import subprocess
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Path to the tool wear prediction script (replace with your actual path)
TOOL_WEAR_SCRIPT = "tool_wear_prediction.py"  # Or a modified version without Streamlit

# Function to run the tool wear prediction script
def run_tool_wear_prediction():
    """
    Runs the tool wear prediction script using subprocess.
    """
    logging.info("Starting tool wear prediction process...")
    try:
        # Use subprocess to run the script in a separate process
        result = subprocess.run(["python", TOOL_WEAR_SCRIPT],
                                capture_output=True, text=True, check=True)

        # Log the output of the script
        logging.info(f"Tool wear prediction script output:\n{result.stdout}")

        # If there are errors, log them
        if result.stderr:
            logging.error(f"Tool wear prediction script errors:\n{result.stderr}")

        logging.info("Tool wear prediction process completed successfully.")

    except subprocess.CalledProcessError as e:
        logging.error(f"Error running tool wear prediction script: {e}")
        logging.error(f"Script output:\n{e.output}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

# Schedule the job (e.g., run every day at 6:00 AM)
schedule.every().day.at("06:00").do(run_tool_wear_prediction)

# Main loop to run the scheduler
if __name__ == "__main__":
    logging.info("Starting scheduler...")
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute