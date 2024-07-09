import time

RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
MAGENTA = '\033[35m'
CYAN = '\033[36m'
WHITE = '\033[37m'
RESET = '\033[0m'

LOG     = BLUE   + "[LOG]      " + RESET
ERROR   = RED    + "[ERROR]    " + RESET
WARNING = YELLOW + "[WARNING]  " + RESET
SUCCESS = GREEN  + "[SUCCESS]  " + RESET

def format_duration(seconds):
    days = seconds // (24 * 3600)
    seconds %= (24 * 3600)
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    
    duration_str = ""
    if days > 0:
        duration_str += f"{days} Day(s) "
    if hours > 0:
        duration_str += f"{hours} Hour(s) "
    if minutes > 0:
        duration_str += f"{minutes} Minute(s) "
    if seconds > 0:
        duration_str += f"{seconds} Second(s)"
    
    return duration_str.strip()