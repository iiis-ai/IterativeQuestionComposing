import datetime
import functools

def log_func_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.datetime.now()
        print(f"{start_time} - start {func.__name__}")
        result = func(*args, **kwargs)
        end_time = datetime.datetime.now()
        print(f"{end_time} - end {func.__name__}")
        return result

    return wrapper
