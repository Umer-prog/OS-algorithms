from Processes import *
from SchedulingAlgo import *

Processes = []
psw = {}
no_of_processes = int(input("Enter Number Of Processes: "))
# Passing list and dictionary by reference
total_exe_time = initialize_processes(no_of_processes, Processes, psw)

# Run the round-robin scheduler
round_robin(Processes, psw, total_exe_time)




