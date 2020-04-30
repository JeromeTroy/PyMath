"""
Methods for getting total times
"""

def get_time_input():
    val = input("Enter the time in hh:mm (24 format)")
    try:
        hours = int(val[:2])
        print(hours)
        print(val[-2:])
        minute = int(val[-2:])
        time = hours * 60 + minutes
    except:
        time = 0
    return time

def print_time(time):
    hours = int(time / 60)
    minute = time % 60
    print(hours, ":", minute)

def get_total_time():
    
    total = 0
    get_new = True
    while get_new:
        print("Enter start time")
        start = get_time_input()

        print("Enter end time")
        end = get_time_input()

        diff = end - start
        total += diff

        if diff == 0:
            get_new = False
    
    print("total time")
    print_time(total)

get_total_time()

        
        
