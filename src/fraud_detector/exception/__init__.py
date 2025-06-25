import sys


"""
Function to get the details of the error
"""


def error_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()  # Get infomrmation of the excpetions
    # Name of the file where an error occured
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno  # Number of the line
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, line_number, str(error)  # Format the message error
    )
    return error_message


"""
Custom exception personlaized class
"""


class CustomException(Exception):
    def __init__(self, error_message, error_details: sys):
        super().__init__(error_message)  # Call the constructor of the base class Excpetion
        self.error_message = error_detail(
            error_message, error_detail=error_detail)  # Generate the detail message

    def __str__(self):
        return self.error_message  # Representation of exception string
