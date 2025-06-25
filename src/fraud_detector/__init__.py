import os
import sys
import logging

"""
Format for the messages logs
"""
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"


"""
Directory for the logs
"""
log_dir = "logs"
log_filepath = os.path.join(log_dir, "running_logs.log")
os.makedirs(log_dir, exist_ok=True)  # Create the directory if not exists.

"""
Basic logger configuration
"""
logging.basicConfig(
    # Minimun level of messages to register (INFO and superiors)
    level=logging.INFO,
    format=logging_str,  # Messages format
    handlers=[
        # File manager to write in a file
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)  # Manager to show in the console
    ]
)


"""
Object logger to use in other parts of the code
"""
logger = logging.getLogger("fraud_detectorLogger")
