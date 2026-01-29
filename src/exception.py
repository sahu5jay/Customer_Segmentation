import sys
from src.logger import logging

def error_message_detail(error, error_detail: sys):
    _,_,exc_tb = error_detail.exc_info()
    # (exception_type, exception_value, traceback_object)
    file_name = exc_tb.tb_frame.f_code.co_filename
    # Python file name where the exception happened.

    error_message = (
        f"error occured in the python script name [{file_name}]"
        f"line number [{exc_tb.tb_lineno}]"
        f"error message [{str(error)}]"
    )
    # {0} → File name {1} → Line number {2} → Error message

    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

        def __str__(self):
            return self.error_message
