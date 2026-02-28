import logging
import os
from datetime import datetime

# Generate a unique log file name with timestamp
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

# Create logs directory path
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)  # ensures "logs" folder exists

# Full path to the log file
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

'''
# Example usage
logging.info("Logging system initialized successfully.")
logging.warning("This is a warning message.")
logging.error("This is an error message.")
'''