from os.path import exists
from pathlib import Path
import logging

#FunctionName: 
#   checkPath
#Input:
#   filePath    :   Path to be checked
#Output:
#   None
#Description:
#   Checks if filePath is a valid path
#Notes:
#   None
def checkPath(filePath):
    if not exists(filePath):
        logging.critical(f"{filePath} is an invalid directory path!")