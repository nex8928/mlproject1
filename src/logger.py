'''
this script sets up a logging configuration that writes 
log messages to a file in a directory named "logs" within
the current working directory, with a filename based on the 
current date and time. The format of each log message includes 
the timestamp,line number, logger name, log level, and message content.
'''

import logging
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(logs_path,exist_ok=True)

LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

'''
This Python script sets up a logging configuration. Let's break down what each part does:

import logging: Imports the Python logging module, which allows for logging messages from your program.
import os: Imports the Python os module, which provides functions for interacting with the operating system.
from datetime import datetime: Imports the datetime class from the datetime module, which is used to get the current date and time.
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log": Defines a variable LOG_FILE which stores the current date and time in the format MM_DD_YYYY_HH_MM_SS.log.
logs_path = os.path.join(os.getcwd(),"logs",LOG_FILE): Constructs the path where the log file will be stored. It joins the current working directory (os.getcwd()), the "logs" directory, and the LOG_FILE.
os.makedirs(logs_path, exist_ok=True): Creates the directory specified by logs_path. The exist_ok=True argument ensures that if the directory already exists, it won't raise an error.
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE): Constructs the full path to the log file by joining logs_path and LOG_FILE.
logging.basicConfig(...): Configures the logging system with the following parameters:
filename=LOG_FILE_PATH: Specifies the file where the log messages will be written.
format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s": Sets the format for log messages. This includes the date and time (%(asctime)s), the line number in the source code (%(lineno)d), the logger name (%(name)s), the log level (%(levelname)s), and the actual message (%(message)s).
level=logging.INFO: Sets the logging level to INFO, meaning only messages with a severity level of INFO or higher will be logged.
Overall, this script sets up a logging configuration that writes log messages to a file in a directory named "logs" within the current working directory, with a filename based on the current date and time. The format of each log message includes the timestamp, line number, logger name, log level, and message content.
'''
