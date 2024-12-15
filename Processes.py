import random


def create_process(PID, arr_t, exe_t, res_info):
    """Generates a Single Process"""
    process_table = {}
    error = 0
    error_message = f"Execution time to big for process {PID}"
    # Check if execution time is within valid range
    if 0 < exe_t <= 10:
        # Generate process with random resource values
        process = [random.randint(1, 100) for _ in range(exe_t)]
        finish_time = arr_t + exe_t

        # Populate process table dictionary
        process_table = {
            "PID": PID,
            "process": process,
            "arrival_time": arr_t,
            "execution_time": exe_t,
            "resource": res_info,
            "finish_time": finish_time
        }
    else:
        error = 1  # Set error flag if exe_t is out of range
    if error == 0:
        return process_table
    else:
        return error_message


def initialize_processes(num_processes, processes, psw):
    """Creates a list of multiple processes, Number is user-defined"""
    total_exe_time = 0
    for process in range(1, num_processes + 1):
        arr_time = int(input(f"Enter Arrival Time for process {process}: "))
        exe_time = int(input(f"Enter Execution Time for process {process}: "))
        res_info = int(input(f"Enter if process {process} Uses any resource: "))
        PID = f"P{process}"
        psw[PID] = 0  # Initialize PSW to 0 for each process
        total_exe_time += exe_time  # Add to the total execution time

        pro = create_process(PID, arr_time, exe_time, res_info)
        processes.append(pro)

    # Sort processes by their arrival time (in-place sorting)
    processes.sort(key=lambda p: p['arrival_time'])

    return total_exe_time

