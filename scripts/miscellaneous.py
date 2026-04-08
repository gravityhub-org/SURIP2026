import time
def get_gps_time_now():
    unix_now = time.time()
    # 2. Convert Unix to GPS seconds
    gps_seconds = (unix_now - 315964800) + 18
    return gps_seconds
