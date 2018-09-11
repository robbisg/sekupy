import time


def get_time():
    """Get the current time and returns a string (fmt: yymmdd_hhmmss)"""
    
    # Time acquisition
    tempo = time.localtime()
    
    datetime = ''
    i = 0
    for elem in tempo[:-3]:
        i = i + 1
        if len(str(elem)) < 2:
            elem = '0'+str(elem)
        if i == 4:
            datetime += '_'
        datetime += str(elem)

    return datetime
