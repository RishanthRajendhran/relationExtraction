from os.path import exists
from pathlib import Path
import logging

#FunctionName: 
#   checkFile
#Input:
#   fileName        :   Path to file to be checked
#   fileExtension   :   Expected file extension
#                       Default: None
#Output:
#   None
#Description:
#   Checks if fileName is a valid file path
#Notes:
#   None
def checkFile(fileName, fileExtension=None):
    if fileExtension:
        if not fileName.endswith(fileExtension):
            logging.critical(f"{fileName} does not have expected file extension {fileExtension}!")
    file_exists = exists(fileName)
    if not file_exists:
        logging.critical(f"{fileName} is an invalid file path!")
    path = Path(fileName)
    if not path.is_file():
        logging.critical(f"{fileName} is not a file!")